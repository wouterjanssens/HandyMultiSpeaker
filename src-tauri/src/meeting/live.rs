//! meetscribe: the live WebVTT transcript (`<meeting>.vtt`) and the `current.vtt` link.
//!
//! While a meeting runs, every turn that can no longer change is appended to the meeting's `.vtt`
//! as a numbered cue (see `meeting_core::vtt` for the format), so people and AI tools can follow
//! the meeting from a file. `current.vtt` in the meetings folder points at the running meeting's
//! file and exists only while a meeting runs. At Stop the file gets `NOTE meeting ended`, and once
//! the final transcript is known it is replaced by the final version.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use meeting_core::{vtt, SpeakerMap, TurnBuilder};

/// Name of the link to the running meeting's live file, in the meetings folder.
pub const CURRENT_NAME: &str = "current.vtt";

pub struct LiveVtt {
    path: PathBuf,
    file: File,
    /// Locked turns already written as cues (cue numbers are 1..=written).
    written: usize,
}

impl LiveVtt {
    /// Creates the live file (`<stem>.vtt`) and points `current.vtt` at it.
    pub fn create(dir: &Path, stem: &str, started_local: &str) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let path = dir.join(format!("{stem}.vtt"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("cannot create {}", path.display()))?;
        file.write_all(vtt::live_header(started_local).as_bytes())?;
        file.flush()?;
        link_current(dir, &path);
        Ok(Self {
            path,
            file,
            written: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends the turns locked since the last call.
    pub fn sync(&mut self, turns: &TurnBuilder, speakers: &SpeakerMap) {
        let locked = turns.locked_count();
        if locked <= self.written {
            return;
        }
        let mut out = String::new();
        for (i, turn) in turns.turns().enumerate().take(locked).skip(self.written) {
            out.push_str(&vtt::cue(i + 1, turn, speakers));
        }
        if self.append(&out) {
            self.written = locked;
        }
    }

    /// Appends a `NOTE` (e.g. a speaker rename).
    pub fn note(&mut self, text: &str) {
        self.append(&vtt::note(text));
    }

    /// Marks the meeting as ended; the file stays until [`finalize`] replaces it.
    pub fn end(mut self, turns: &TurnBuilder, speakers: &SpeakerMap) -> PathBuf {
        self.sync(turns, speakers);
        self.append(&format!("{}\n", vtt::ENDED_MARKER));
        self.path
    }

    /// One whole write, flushed, so a reader never sees half a cue.
    fn append(&mut self, text: &str) -> bool {
        match self
            .file
            .write_all(text.as_bytes())
            .and_then(|_| self.file.flush())
        {
            Ok(()) => true,
            Err(e) => {
                log::warn!("Cannot append to {}: {e}", self.path.display());
                false
            }
        }
    }
}

/// Replaces the live file with the final transcript (atomically) and removes `current.vtt`.
pub fn finalize(path: &Path, document: &str) {
    let tmp = path.with_extension("vtt.tmp");
    let result = std::fs::write(&tmp, document).and_then(|_| std::fs::rename(&tmp, path));
    if let Err(e) = result {
        log::warn!("Cannot write the final {}: {e}", path.display());
        let _ = std::fs::remove_file(&tmp);
    }
    if let Some(dir) = path.parent() {
        remove_current(dir);
    }
}

/// Removes a `current.vtt` left behind when the app quit or crashed mid-meeting.
pub fn remove_stale_current(dir: &Path) {
    remove_current(dir);
}

fn remove_current(dir: &Path) {
    let link = dir.join(CURRENT_NAME);
    if link.symlink_metadata().is_ok() {
        if let Err(e) = std::fs::remove_file(&link) {
            log::warn!("Cannot remove {}: {e}", link.display());
        }
    }
}

fn link_current(dir: &Path, target: &Path) {
    remove_current(dir);
    let link = dir.join(CURRENT_NAME);
    // relative, so the folder can be moved or synced
    let name = target.file_name().map(PathBuf::from).unwrap_or_default();
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(&name, &link);
    // Windows symlinks need extra rights; a hard link is the same file and grows with it.
    #[cfg(not(unix))]
    let result = std::fs::hard_link(target, &link);
    #[cfg(not(unix))]
    let _ = name;
    if let Err(e) = result {
        log::warn!("Cannot create {}: {e}", link.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meeting_core::{SpeakerActivity, TimedWord};

    fn dir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "meetscribe-live-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn finished(texts: &[(&str, u64)]) -> (TurnBuilder, SpeakerMap) {
        let mut turns = TurnBuilder::new(Default::default());
        turns.add_words(texts.iter().map(|(t, ms)| TimedWord {
            text: t.to_string(),
            t0_ms: *ms,
            t1_ms: ms + 400,
        }));
        let _ = turns.finish(&SpeakerActivity::new(2, 80));
        (turns, SpeakerMap::new())
    }

    #[test]
    fn live_file_is_appended_linked_and_finalized() {
        let d = dir();
        let mut live = LiveVtt::create(&d, "2026-11-19_1000_meeting", "2026-11-19 10:00").unwrap();
        let path = live.path().to_path_buf();
        assert!(d.join(CURRENT_NAME).exists());

        let (turns, speakers) = finished(&[("hallo", 0), ("daar", 20_000)]);
        live.sync(&turns, &speakers);
        live.sync(&turns, &speakers); // nothing new: no duplicate cues
        live.note("Speaker 1 is now called Wouter.");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("WEBVTT\n\nNOTE\nMeetScribe live transcript."));
        assert_eq!(text.matches("\n1\n").count(), 1);
        assert!(text.contains("\n2\n"));
        assert!(text.ends_with("NOTE Speaker 1 is now called Wouter.\n\n"));

        let path = live.end(&turns, &speakers);
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .ends_with("NOTE meeting ended\n"));

        finalize(&path, "WEBVTT\n\nNOTE final\n");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "WEBVTT\n\nNOTE final\n"
        );
        assert!(d.join(CURRENT_NAME).symlink_metadata().is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
