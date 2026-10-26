//! Meeting output: a Markdown transcript for people and a word-level JSON sidecar for tools
//! (e.g. a later LLM summary step).

use serde::Serialize;

use crate::speakers::SpeakerMap;
use crate::turns::Turn;

/// Metadata for the output header and file names.
#[derive(Debug, Clone, Serialize)]
pub struct MeetingInfo {
    /// Local start time, already formatted as `YYYY-MM-DD HH:MM`.
    pub started_local: String,
    pub duration_ms: u64,
    pub asr_model: String,
    pub diarizer_model: String,
    pub dropped_audio_ms: u64,
}

/// `HH:MM:SS` for a millisecond offset.
pub fn hms(ms: u64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn duration_label(ms: u64) -> String {
    let min = (ms + 30_000) / 60_000;
    if min >= 60 {
        format!("{} h {} min", min / 60, min % 60)
    } else {
        format!("{min} min")
    }
}

/// Speakers in order of first appearance in the transcript.
fn speakers_in_order<'a>(turns: impl IntoIterator<Item = &'a Turn>) -> Vec<Option<u8>> {
    let mut out = Vec::new();
    for t in turns {
        if !out.contains(&t.speaker) {
            out.push(t.speaker);
        }
    }
    out
}

pub fn to_markdown<'a>(
    info: &MeetingInfo,
    turns: impl IntoIterator<Item = &'a Turn> + Clone,
    speakers: &SpeakerMap,
) -> String {
    let mut md = format!(
        "# Meeting – {} ({})\n",
        info.started_local,
        duration_label(info.duration_ms)
    );
    let labels: Vec<String> = speakers_in_order(turns.clone())
        .into_iter()
        .map(|s| speakers.label(s))
        .collect();
    md.push_str(&format!("Speakers: {}\n\n", labels.join(", ")));
    for t in turns {
        md.push_str(&format!(
            "**{}** [{}]: {}\n\n",
            speakers.label(t.speaker),
            hms(t.start_ms),
            t.text
        ));
    }
    md
}

#[derive(Serialize)]
struct JsonWord<'a> {
    start_ms: u64,
    end_ms: u64,
    text: &'a str,
    speaker: Option<u8>,
    confidence: f32,
}

#[derive(Serialize)]
struct JsonTurn<'a> {
    speaker: Option<u8>,
    speaker_label: String,
    start_ms: u64,
    end_ms: u64,
    text: &'a str,
    words: Vec<JsonWord<'a>>,
}

#[derive(Serialize)]
struct JsonSpeaker {
    id: Option<u8>,
    label: String,
    custom_name: Option<String>,
}

#[derive(Serialize)]
struct JsonMeeting<'a> {
    format: &'static str,
    info: &'a MeetingInfo,
    speakers: Vec<JsonSpeaker>,
    turns: Vec<JsonTurn<'a>>,
}

pub fn to_json<'a>(
    info: &MeetingInfo,
    turns: impl IntoIterator<Item = &'a Turn> + Clone,
    speakers: &SpeakerMap,
) -> serde_json::Result<String> {
    let doc = JsonMeeting {
        format: "meetscribe.meeting.v1",
        info,
        speakers: speakers_in_order(turns.clone())
            .into_iter()
            .map(|s| JsonSpeaker {
                id: s,
                label: speakers.label(s),
                custom_name: s.and_then(|s| speakers.custom_name(s)).map(str::to_string),
            })
            .collect(),
        turns: turns
            .into_iter()
            .map(|t| JsonTurn {
                speaker: t.speaker,
                speaker_label: speakers.label(t.speaker),
                start_ms: t.start_ms,
                end_ms: t.end_ms,
                text: &t.text,
                words: t
                    .words
                    .iter()
                    .map(|w| JsonWord {
                        start_ms: w.word.t0_ms,
                        end_ms: w.word.t1_ms,
                        text: &w.word.text,
                        speaker: w.speaker,
                        confidence: w.confidence,
                    })
                    .collect(),
            })
            .collect(),
    };
    serde_json::to_string_pretty(&doc)
}

/// File stem `YYYY-MM-DD_HHMM_meeting` from a `YYYY-MM-DD HH:MM` start label.
pub fn file_stem(started_local: &str) -> String {
    let compact: String = started_local.replace(':', "").replace(' ', "_");
    format!("{compact}_meeting")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::{AttributedWord, TimedWord};

    fn turn(id: u32, speaker: Option<u8>, start_ms: u64, text: &str) -> Turn {
        Turn {
            id,
            speaker,
            start_ms,
            end_ms: start_ms + 1_000,
            text: text.into(),
            locked: true,
            words: vec![AttributedWord {
                word: TimedWord::new(start_ms, start_ms + 1_000, text),
                speaker,
                confidence: 0.9,
                tentative: false,
            }],
        }
    }

    fn info() -> MeetingInfo {
        MeetingInfo {
            started_local: "2026-10-26 14:00".into(),
            duration_ms: 52 * 60_000,
            asr_model: "parakeet-tdt-0.6b-v3".into(),
            diarizer_model: "Nemotron-3-Diarization".into(),
            dropped_audio_ms: 0,
        }
    }

    #[test]
    fn markdown_matches_spec_format() {
        let mut speakers = SpeakerMap::new();
        for id in [0, 1, 2] {
            speakers.observe(id);
        }
        speakers.rename(0, "Jan");
        speakers.rename(2, "Wouter");
        let turns = [
            turn(0, Some(0), 4_000, "Goeiemiddag."),
            turn(1, Some(1), 12_000, "Dag allemaal."),
            turn(2, Some(2), 31_000, "Zullen we beginnen?"),
        ];
        let md = to_markdown(&info(), turns.iter(), &speakers);
        assert_eq!(
            md,
            "# Meeting – 2026-10-26 14:00 (52 min)\nSpeakers: Jan, Speaker 2, Wouter\n\n\
             **Jan** [00:00:04]: Goeiemiddag.\n\n\
             **Speaker 2** [00:00:12]: Dag allemaal.\n\n\
             **Wouter** [00:00:31]: Zullen we beginnen?\n\n"
        );
    }

    #[test]
    fn json_has_words_and_labels() {
        let mut speakers = SpeakerMap::new();
        speakers.rename(1, "Jan");
        let turns = [turn(0, Some(1), 0, "hallo"), turn(1, None, 2_000, "eh")];
        let json = to_json(&info(), turns.iter(), &speakers).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["format"], "meetscribe.meeting.v1");
        assert_eq!(v["turns"][0]["speaker_label"], "Jan");
        assert_eq!(v["turns"][0]["words"][0]["text"], "hallo");
        assert_eq!(v["turns"][1]["speaker_label"], "Unknown");
        assert_eq!(v["speakers"][0]["custom_name"], "Jan");
    }

    #[test]
    fn helpers() {
        assert_eq!(hms(3_725_000), "01:02:05");
        assert_eq!(duration_label(75 * 60_000), "1 h 15 min");
        assert_eq!(file_stem("2026-10-26 14:00"), "2026-10-26_1400_meeting");
    }
}
