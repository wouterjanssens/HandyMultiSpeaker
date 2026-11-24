//! Keeps OpenAI speaker labels consistent across a meeting.
//!
//! `gpt-4o-transcribe-diarize` labels each request on its own (`A`, `B`, …) unless it is given
//! known voices: up to four names with a 2–10 s reference clip each. The tracker enrols a new
//! voice from the first long-enough stretch of an unknown speaker, so the next utterances come
//! back under that voice's name. Unknown speakers that can't be enrolled (too short, or four
//! voices already known) get no label and are attributed by the aligner (nearest speaker).

use super::client::{wav_data_url, DiarSegment, KnownSpeaker};

const SAMPLE_RATE: f64 = 16_000.0;
/// API limits for reference clips and the number of known speakers.
const MIN_REF_SECS: f64 = 2.0;
const MAX_REF_SECS: f64 = 9.5;
pub const MAX_VOICES: usize = 4;
/// A reference shorter than this is replaced by a longer clip of the same voice when one comes.
const GOOD_REF_SECS: f64 = 6.0;

struct Voice {
    name: String,
    slot: u8,
    data_url: String,
    ref_secs: f64,
}

/// One speaker's stretch inside an utterance (ms from the utterance start).
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub speaker: Option<u8>,
    pub t0_ms: u64,
    pub t1_ms: u64,
    /// The diarizer's own text (only used if re-transcribing the run fails).
    pub text: String,
}

#[derive(Default)]
pub struct SpeakerTracker {
    voices: Vec<Voice>,
}

impl SpeakerTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn known(&self) -> Vec<KnownSpeaker> {
        self.voices
            .iter()
            .map(|v| KnownSpeaker {
                name: v.name.clone(),
                data_url: v.data_url.clone(),
            })
            .collect()
    }

    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Maps one response's segments to speaker slots (enrolling voices from `pcm`, the request
    /// audio) and merges consecutive same-speaker segments into runs.
    pub fn resolve(&mut self, segments: &[DiarSegment], pcm: &[f32]) -> Vec<Run> {
        let clip = |start: f64, end: f64| -> (Vec<f32>, f64) {
            let (mut s, mut e) = (start.max(0.0), end);
            if e - s > MAX_REF_SECS {
                let mid = (s + e) / 2.0;
                s = mid - MAX_REF_SECS / 2.0;
                e = mid + MAX_REF_SECS / 2.0;
            }
            let a = ((s * SAMPLE_RATE) as usize).min(pcm.len());
            let b = ((e * SAMPLE_RATE) as usize).min(pcm.len());
            (pcm[a..b].to_vec(), (b - a) as f64 / SAMPLE_RATE)
        };

        let segments: Vec<&DiarSegment> = segments.iter().filter(|s| s.end > s.start).collect();
        let mut fresh: Vec<(String, Option<u8>)> = Vec::new();
        let mut slots: Vec<Option<u8>> = Vec::with_capacity(segments.len());
        for seg in &segments {
            if let Some(i) = self.voices.iter().position(|v| v.name == seg.speaker) {
                // a longer clip makes a better reference
                let len = (seg.end - seg.start).min(MAX_REF_SECS);
                if self.voices[i].ref_secs < GOOD_REF_SECS && len > self.voices[i].ref_secs + 1.5 {
                    let (pcm, secs) = clip(seg.start, seg.end);
                    if secs >= MIN_REF_SECS {
                        self.voices[i].data_url = wav_data_url(&pcm);
                        self.voices[i].ref_secs = secs;
                    }
                }
                slots.push(Some(self.voices[i].slot));
                continue;
            }
            if let Some((_, slot)) = fresh.iter().find(|(l, _)| *l == seg.speaker) {
                slots.push(*slot);
                continue;
            }
            // first segment of an unknown label in this response: enrol from all of its
            // segments joined (one short phrase is often split into sub-2 s pieces)
            let mut slot = None;
            if self.voices.len() < MAX_VOICES {
                let mut pcm = Vec::new();
                for s in segments.iter().filter(|s| s.speaker == seg.speaker) {
                    let room = MAX_REF_SECS - pcm.len() as f64 / SAMPLE_RATE;
                    if room <= 0.0 {
                        break;
                    }
                    pcm.extend(clip(s.start, s.end.min(s.start + room)).0);
                }
                let secs = pcm.len() as f64 / SAMPLE_RATE;
                if secs >= MIN_REF_SECS {
                    let s = self.voices.len() as u8;
                    self.voices.push(Voice {
                        name: format!("S{}", s + 1),
                        slot: s,
                        data_url: wav_data_url(&pcm),
                        ref_secs: secs,
                    });
                    slot = Some(s);
                }
            }
            fresh.push((seg.speaker.clone(), slot));
            slots.push(slot);
        }

        let mut runs: Vec<Run> = Vec::new();
        for (seg, slot) in segments.iter().zip(slots) {
            let t0 = (seg.start * 1000.0).max(0.0) as u64;
            let t1 = (seg.end * 1000.0) as u64;
            match runs.last_mut() {
                Some(r) if r.speaker == slot => {
                    r.t1_ms = r.t1_ms.max(t1);
                    r.text.push_str(&seg.text);
                }
                _ => runs.push(Run {
                    speaker: slot,
                    t0_ms: t0,
                    t1_ms: t1,
                    text: seg.text.clone(),
                }),
            }
        }
        runs
    }
}

/// Audio span to re-transcribe for each run: gaps are split at their midpoint, the first run
/// starts at 0 and the last ends at `total_ms`, so no speech is lost between runs.
pub fn slice_bounds(runs: &[Run], total_ms: u64) -> Vec<(u64, u64)> {
    (0..runs.len())
        .map(|i| {
            let start = if i == 0 {
                0
            } else {
                (runs[i - 1].t1_ms.min(runs[i].t0_ms) + runs[i].t0_ms) / 2
            };
            let end = if i + 1 == runs.len() {
                total_ms
            } else {
                (runs[i].t1_ms + runs[i + 1].t0_ms.max(runs[i].t1_ms)) / 2
            };
            (
                start.min(total_ms),
                end.min(total_ms).max(start.min(total_ms)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(speaker: &str, start: f64, end: f64) -> DiarSegment {
        DiarSegment {
            speaker: speaker.into(),
            start,
            end,
            text: format!(" {speaker}"),
        }
    }

    fn silence(secs: f64) -> Vec<f32> {
        vec![0.0; (secs * SAMPLE_RATE) as usize]
    }

    #[test]
    fn enrols_long_unknown_speakers_and_reuses_them_by_name() {
        let mut t = SpeakerTracker::new();
        let pcm = silence(12.0);
        let runs = t.resolve(&[seg("A", 0.0, 4.0), seg("B", 4.5, 5.0)], &pcm);
        // A long enough → S1; B too short → unlabelled
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].speaker, Some(0));
        assert_eq!(runs[1].speaker, None);
        assert_eq!(t.voice_count(), 1);
        assert_eq!(t.known()[0].name, "S1");

        // next request: S1 comes back by name, a new long speaker becomes slot 1
        let runs = t.resolve(&[seg("S1", 0.0, 1.0), seg("A", 1.2, 4.0)], &pcm);
        assert_eq!(
            runs.iter().map(|r| r.speaker).collect::<Vec<_>>(),
            vec![Some(0), Some(1)]
        );
        assert_eq!(t.voice_count(), 2);
    }

    #[test]
    fn short_pieces_of_one_speaker_are_joined_into_a_reference() {
        let mut t = SpeakerTracker::new();
        let pcm = silence(6.0);
        let runs = t.resolve(
            &[seg("A", 0.0, 1.1), seg("A", 1.4, 2.3), seg("A", 2.5, 3.4)],
            &pcm,
        );
        assert_eq!(runs[0].speaker, Some(0));
        assert_eq!(t.voice_count(), 1);
    }

    #[test]
    fn same_speaker_segments_merge_into_one_run() {
        let mut t = SpeakerTracker::new();
        let pcm = silence(10.0);
        let runs = t.resolve(
            &[seg("A", 0.0, 1.0), seg("A", 1.2, 3.5), seg("A", 4.0, 5.0)],
            &pcm,
        );
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].t0_ms, runs[0].t1_ms), (0, 5000));
        assert_eq!(runs[0].speaker, Some(0));
    }

    #[test]
    fn at_most_four_voices() {
        let mut t = SpeakerTracker::new();
        let pcm = silence(30.0);
        for _ in 0..6 {
            t.resolve(&[seg("A", 0.0, 3.0)], &pcm);
        }
        assert_eq!(t.voice_count(), MAX_VOICES);
        let runs = t.resolve(&[seg("A", 0.0, 3.0)], &pcm);
        assert_eq!(runs[0].speaker, None);
    }

    #[test]
    fn slices_cover_the_whole_utterance() {
        let r = |t0, t1| Run {
            speaker: None,
            t0_ms: t0,
            t1_ms: t1,
            text: String::new(),
        };
        let runs = [r(300, 2000), r(2400, 5000), r(5000, 7000)];
        assert_eq!(
            slice_bounds(&runs, 8000),
            vec![(0, 2200), (2200, 5000), (5000, 8000)]
        );
    }
}
