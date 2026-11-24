//! WebVTT transcript, written live while a meeting runs.
//!
//! The live file is only ever appended to, so a reader (a person with `tail -f`, or an AI that
//! remembers the last cue number it read) never sees text change under it:
//! - a header `NOTE` says the file is live and how it ends;
//! - each finished (locked) turn becomes one cue, numbered 1, 2, 3… with `<v Speaker>` voice tags;
//! - a speaker rename is an appended `NOTE`, since earlier cues are not rewritten;
//! - Stop appends `NOTE meeting ended`, then the file is replaced once by the final transcript
//!   ([`final_document`]), which ends with `NOTE final`.

use crate::speakers::SpeakerMap;
use crate::turns::Turn;

/// Last line of a live file whose meeting has ended (before the final rewrite).
pub const ENDED_MARKER: &str = "NOTE meeting ended";
/// Last line of the final file.
pub const FINAL_MARKER: &str = "NOTE final";

/// `HH:MM:SS.mmm` for a millisecond offset from the start of the meeting.
pub fn timestamp(ms: u64) -> String {
    let s = ms / 1000;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        s / 3600,
        (s / 60) % 60,
        s % 60,
        ms % 1000
    )
}

/// Text inside a cue may not contain `-->` and must escape `&` and `<`.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace("-->", "--&gt;")
}

/// A `NOTE` body may not contain a blank line or `-->`.
fn note_text(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .replace("-->", "->")
}

/// One `NOTE` block, followed by the blank line that ends it.
pub fn note(text: &str) -> String {
    format!("NOTE {}\n\n", note_text(text))
}

/// The header of a live file.
pub fn live_header(started_local: &str) -> String {
    format!(
        "WEBVTT\n\nNOTE\nMeetScribe live transcript. Meeting started {started_local}.\n\
         This file grows while the meeting runs: each finished line is added at the end as a\n\
         numbered cue, and nothing above is ever changed. Times are from the start of the meeting.\n\
         A speaker rename is added as a NOTE and applies to earlier cues too.\n\
         The meeting is over when the file ends with \"{ENDED_MARKER}\"; the file is then\n\
         replaced once by the final transcript, which ends with \"{FINAL_MARKER}\".\n\n"
    )
}

/// One cue for `turn`, numbered `number`, with the speaker's current label.
pub fn cue(number: usize, turn: &Turn, speakers: &SpeakerMap) -> String {
    format!(
        "{number}\n{} --> {}\n<v {}>{}\n\n",
        timestamp(turn.start_ms),
        timestamp(turn.end_ms.max(turn.start_ms)),
        escape(&speakers.label(turn.speaker)),
        escape(turn.text.trim())
    )
}

/// The final transcript: every turn with its final speaker name, ending with [`FINAL_MARKER`].
pub fn final_document<'a>(
    started_local: &str,
    turns: impl IntoIterator<Item = &'a Turn>,
    speakers: &SpeakerMap,
    markdown_name: Option<&str>,
) -> String {
    let mut out = format!(
        "WEBVTT\n\nNOTE\nMeetScribe transcript. Meeting started {started_local}.\n\
         Final version: this file no longer changes. Times are from the start of the meeting.\n"
    );
    if let Some(md) = markdown_name {
        out.push_str(&format!("Same transcript as Markdown: {md}\n"));
    }
    out.push('\n');
    for (i, turn) in turns.into_iter().enumerate() {
        out.push_str(&cue(i + 1, turn, speakers));
    }
    out.push_str(FINAL_MARKER);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(id: u32, speaker: Option<u8>, start_ms: u64, text: &str) -> Turn {
        Turn {
            id,
            speaker,
            start_ms,
            end_ms: start_ms + 2_500,
            text: text.into(),
            locked: true,
            words: vec![],
        }
    }

    #[test]
    fn timestamps_have_hours_and_millis() {
        assert_eq!(timestamp(0), "00:00:00.000");
        assert_eq!(timestamp(3_723_045), "01:02:03.045");
    }

    #[test]
    fn cue_uses_voice_tag_and_escapes() {
        let mut speakers = SpeakerMap::new();
        speakers.observe(3);
        speakers.rename(3, "Wouter");
        let c = cue(7, &turn(6, Some(3), 1_200, "a < b & c --> d"), &speakers);
        assert_eq!(
            c,
            "7\n00:00:01.200 --> 00:00:03.700\n<v Wouter>a &lt; b &amp; c --&gt; d\n\n"
        );
    }

    #[test]
    fn note_never_contains_a_blank_line() {
        assert_eq!(note("one\n\ntwo"), "NOTE one\ntwo\n\n");
    }

    #[test]
    fn final_document_numbers_cues_and_ends_final() {
        let mut speakers = SpeakerMap::new();
        speakers.observe(0);
        speakers.observe(1);
        let turns = [turn(0, Some(0), 0, "Hallo"), turn(1, Some(1), 3_000, "Hi")];
        let doc = final_document("2026-11-19 10:00", &turns, &speakers, Some("x.md"));
        assert!(doc.starts_with("WEBVTT\n\nNOTE\n"));
        assert!(doc.contains("1\n00:00:00.000 --> 00:00:02.500\n<v Speaker 1>Hallo\n\n"));
        assert!(doc.contains("2\n00:00:03.000 --> 00:00:05.500\n<v Speaker 2>Hi\n\n"));
        assert!(doc.ends_with("NOTE final\n"));
    }
}
