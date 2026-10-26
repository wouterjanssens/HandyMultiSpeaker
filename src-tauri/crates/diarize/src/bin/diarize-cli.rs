//! Streams a 16 kHz mono WAV through the native diarizer the way the app does (512-sample
//! pushes), prints RTTM to stdout and timing to stderr, and optionally dumps the raw frame
//! probabilities for parity checks against the Python reference
//! (`research/compare_native.py`).
//!
//! diarize-cli <file.wav> --model <Nemotron-3-Diarization.q8_0.gguf>
//!             [--profile low_latency|very_low_latency|ultra_low_latency|p5s] [--cpu]
//!             [--dump-probs out.f32]

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use diarize::{NativeConfig, NativeDiarizer};
use meeting_core::{SpeakerActivity, StreamingDiarizer};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut wav = None;
    let mut model = None;
    let mut profile = "low_latency".to_string();
    let mut cpu = false;
    let mut dump = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            "--profile" => profile = args.next().context("--profile needs a value")?,
            "--cpu" => cpu = true,
            "--dump-probs" => dump = args.next().map(PathBuf::from),
            _ if wav.is_none() => wav = Some(PathBuf::from(a)),
            other => bail!("unexpected argument {other}"),
        }
    }
    let wav = wav.context(
        "usage: diarize-cli <file.wav> --model <gguf> [--profile P] [--cpu] [--dump-probs F]",
    )?;
    let model = model.context("--model is required")?;

    let mut reader =
        hound::WavReader::open(&wav).with_context(|| format!("open {}", wav.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        bail!(
            "expected 16 kHz mono, got {} Hz x {}",
            spec.sample_rate,
            spec.channels
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    let audio_s = samples.len() as f64 / 16_000.0;

    let mut cfg = NativeConfig::for_profile(&profile);
    if cpu {
        cfg.gpu = None;
    }
    let t_load = Instant::now();
    let mut d = NativeDiarizer::new(&model, &cfg)?;
    eprintln!(
        "loaded {} in {:.2}s",
        d.describe(),
        t_load.elapsed().as_secs_f64()
    );

    let mut activity = SpeakerActivity::new(d.num_speakers(), d.frame_ms());
    let t0 = Instant::now();
    let mut push_ms = Vec::new();
    for chunk in samples.chunks(512) {
        let t = Instant::now();
        let u = d.push(chunk)?;
        push_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        activity.write(u.start_frame, &u.probs);
    }
    let u = d.finish()?;
    activity.write(u.start_frame, &u.probs);
    let compute = t0.elapsed().as_secs_f64();
    push_ms.sort_by(|a, b| a.total_cmp(b));
    eprintln!(
        "audio {audio_s:.1}s, compute {compute:.2}s, RTF {:.4}, frames {}, push p50 {:.2} ms / p99 {:.2} ms / max {:.1} ms",
        compute / audio_s,
        activity.end_frame(),
        push_ms[push_ms.len() / 2],
        push_ms[push_ms.len() * 99 / 100],
        push_ms.last().copied().unwrap_or(0.0),
    );

    if let Some(path) = dump {
        let n = activity.num_speakers();
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for frame in 0..activity.end_frame() {
            for s in 0..n {
                f.write_all(&activity.prob(frame, s).to_le_bytes())?;
            }
        }
        eprintln!(
            "wrote {} x {n} probabilities to {}",
            activity.end_frame(),
            path.display()
        );
    }

    let uri = wav.file_stem().unwrap_or_default().to_string_lossy();
    for s in d.segments()? {
        println!(
            "SPEAKER {uri} 1 {:.2} {:.2} <NA> <NA> spk{} <NA> <NA>",
            s.start_time,
            s.end_time - s.start_time,
            s.speaker - 1
        );
    }
    Ok(())
}
