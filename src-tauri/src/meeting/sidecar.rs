//! TEMPORARY diarizer backend: the Hugging Face reference implementation of Nemotron 3
//! Diarization running in a Python child process (`research/diarize_server.py`).
//!
//! This exists so meeting mode can be exercised end to end before the native NeMo-Speech.cpp
//! backend is in place. It needs the `research/` uv environment, so it only works in a dev
//! checkout. Audio goes to the child over a pipe and never leaves the machine.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use meeting_core::{DiarUpdate, StreamingDiarizer};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Message {
    Ready {
        num_speakers: usize,
        frame_ms: u64,
        latency_ms: u64,
        describe: String,
    },
    Frames {
        start_frame: u64,
        num_frames: usize,
        probs: Vec<f32>,
        finalized_until: u64,
    },
    Done,
}

pub struct PythonSidecarDiarizer {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<Result<Message>>,
    num_speakers: usize,
    frame_ms: u64,
    latency_ms: u64,
    describe: String,
}

/// The `research/` folder of this checkout (override with `MEETSCRIBE_RESEARCH_DIR`).
pub fn research_dir() -> PathBuf {
    std::env::var_os("MEETSCRIBE_RESEARCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../research"))
}

/// `uv`, looked up in places a GUI-launched app's minimal PATH misses.
fn find_uv() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).map(|d| d.join("uv")).collect())
        .unwrap_or_default();
    candidates.push("/opt/homebrew/bin/uv".into());
    candidates.push("/usr/local/bin/uv".into());
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(Path::new(&home).join(".local/bin/uv"));
        candidates.push(Path::new(&home).join(".cargo/bin/uv"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

impl PythonSidecarDiarizer {
    /// Starts the sidecar and waits for the model to load (can take a while on first use, when
    /// the weights are downloaded into the Hugging Face cache).
    pub fn spawn(profile: &str, device: &str, ready_timeout: Duration) -> Result<Self> {
        let dir = research_dir();
        let script = dir.join("diarize_server.py");
        if !script.is_file() {
            bail!(
                "diarizer sidecar not found at {} (dev checkouts only; set MEETSCRIBE_RESEARCH_DIR)",
                script.display()
            );
        }
        let uv = find_uv()
            .ok_or_else(|| anyhow!("`uv` not found; install it to use the Python diarizer"))?;
        let mut child = Command::new(uv)
            .current_dir(&dir)
            .args(["run", "--project", ".", "python", "diarize_server.py"])
            .args(["--profile", profile, "--device", device])
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to start the diarizer sidecar")?;

        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        std::thread::Builder::new()
            .name("meeting-diar-stderr".into())
            .spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    log::debug!("[diarize_server] {line}");
                }
            })?;
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("meeting-diar-stdout".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let msg = line
                        .map_err(anyhow::Error::from)
                        .and_then(|l| serde_json::from_str::<Message>(&l).map_err(Into::into));
                    let failed = msg.is_err();
                    if tx.send(msg).is_err() || failed {
                        break;
                    }
                }
            })?;

        let stdin = child.stdin.take();
        let mut me = Self {
            child,
            stdin,
            rx,
            num_speakers: 0,
            frame_ms: 0,
            latency_ms: 0,
            describe: String::new(),
        };
        match me.rx.recv_timeout(ready_timeout) {
            Ok(Ok(Message::Ready {
                num_speakers,
                frame_ms,
                latency_ms,
                describe,
            })) => {
                me.num_speakers = num_speakers;
                me.frame_ms = frame_ms;
                me.latency_ms = latency_ms;
                me.describe = describe;
                Ok(me)
            }
            Ok(Ok(other)) => bail!("diarizer sidecar sent {other:?} before ready"),
            Ok(Err(e)) => Err(e.context("diarizer sidecar failed to start")),
            Err(_) => bail!("diarizer sidecar did not become ready in {ready_timeout:?}"),
        }
    }

    fn write_block(&mut self, samples: &[f32]) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("diarizer already finished"))?;
        let mut buf = Vec::with_capacity(4 + samples.len() * 4);
        buf.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        for s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        stdin
            .write_all(&buf)
            .context("diarizer sidecar pipe closed")?;
        stdin.flush()?;
        Ok(())
    }

    /// Merges consecutive frame messages into one update.
    fn absorb(&self, update: &mut DiarUpdate, msg: Message) -> Result<bool> {
        match msg {
            Message::Frames {
                start_frame,
                num_frames,
                probs,
                finalized_until,
            } => {
                if probs.len() != num_frames * self.num_speakers {
                    bail!("diarizer sidecar sent a malformed frame block");
                }
                if update.probs.is_empty() {
                    update.start_frame = start_frame;
                }
                // blocks are contiguous; anything else is a protocol error
                let expected = update.start_frame + (update.probs.len() / self.num_speakers) as u64;
                if start_frame != expected {
                    bail!("diarizer sidecar frames out of order ({start_frame} != {expected})");
                }
                update.probs.extend_from_slice(&probs);
                update.finalized_until = finalized_until;
                Ok(false)
            }
            Message::Done => Ok(true),
            Message::Ready { .. } => bail!("unexpected second ready message"),
        }
    }
}

impl StreamingDiarizer for PythonSidecarDiarizer {
    fn num_speakers(&self) -> usize {
        self.num_speakers
    }

    fn frame_ms(&self) -> u64 {
        self.frame_ms
    }

    fn latency_ms(&self) -> u64 {
        self.latency_ms
    }

    fn push(&mut self, samples: &[f32]) -> Result<DiarUpdate> {
        if !samples.is_empty() {
            self.write_block(samples)?;
        }
        let mut update = DiarUpdate::default();
        while let Ok(msg) = self.rx.try_recv() {
            self.absorb(&mut update, msg?)?;
        }
        Ok(update)
    }

    fn finish(&mut self) -> Result<DiarUpdate> {
        self.write_block(&[])?; // n == 0: finish
        self.stdin = None;
        let mut update = DiarUpdate::default();
        loop {
            match self.rx.recv_timeout(Duration::from_secs(60)) {
                Ok(msg) => {
                    if self.absorb(&mut update, msg?)? {
                        return Ok(update);
                    }
                }
                Err(RecvTimeoutError::Timeout) => bail!("diarizer sidecar did not finish in time"),
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("diarizer sidecar exited before finishing")
                }
            }
        }
    }

    fn describe(&self) -> String {
        self.describe.clone()
    }
}

impl Drop for PythonSidecarDiarizer {
    fn drop(&mut self) {
        self.stdin = None;
        // give it a moment to exit on EOF, then make sure it is gone
        for _ in 0..20 {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
