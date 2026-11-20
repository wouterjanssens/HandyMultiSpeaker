//! meetscribe: end-of-meeting speaker pass on OpenAI.
//!
//! Per-utterance diarization (live) only hears a few seconds at a time and confuses similar
//! voices. After the meeting the whole recording is diarized with full context, in chunks that
//! fit the upload limit; chunks after the first get the first chunk's voices as known speakers so
//! labels stay consistent. The `gpt-transcribe` text from the live pass is kept; only speakers
//! change, plus spans the full pass shows to hold several speakers are re-transcribed per speaker.
//! New speaker ids are mapped onto the live ones by overlap, so names given during the meeting
//! survive.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use meeting_core::{AttributedWord, SpeakerActivity, TimedWord};

use crate::openai::client::{wav_data_url, DiarSegment, KnownSpeaker};
use crate::openai::speakers::{slice_bounds, Run};
use crate::openai::{Client, OpenAiSettings};

const SLOTS: usize = 8;
const FRAME_MS: u64 = 10;
const RATE: u64 = 16;
/// 10 min of 16 kHz PCM16 WAV ≈ 19 MB, under the 25 MB upload limit.
const CHUNK_MS: u64 = 600_000;
/// A speaker change inside a live span is only acted on when each part is at least this long.
const MIN_PART_MS: u64 = 1_000;

/// A speaker stretch on the meeting timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub t0_ms: u64,
    pub t1_ms: u64,
    pub slot: u8,
}

/// Reads ranges of the meeting WAV (16 kHz mono PCM16).
struct Wav {
    path: std::path::PathBuf,
    len: u64,
}

impl Wav {
    fn open(path: &Path) -> Result<Self> {
        let r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            len: r.duration() as u64,
        })
    }

    fn len_ms(&self) -> u64 {
        self.len / RATE
    }

    fn read(&self, t0_ms: u64, t1_ms: u64) -> Result<Vec<f32>> {
        let mut r = hound::WavReader::open(&self.path)?;
        let a = (t0_ms * RATE).min(self.len);
        let b = (t1_ms * RATE).min(self.len);
        r.seek(a as u32)?;
        Ok(r.samples::<i16>()
            .take((b - a) as usize)
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect())
    }
}

/// Chunks of at most `max_ms`, cut in the longest pause between utterances in the last third of
/// each chunk (or hard at `max_ms` when there is none).
pub fn plan_chunks(total_ms: u64, utterances: &[(u64, u64)], max_ms: u64) -> Vec<(u64, u64)> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while total_ms - start > max_ms {
        let limit = start + max_ms;
        let earliest = start + max_ms * 2 / 3;
        let cut = utterances
            .windows(2)
            .filter(|w| w[0].1 >= earliest && w[1].0 <= limit && w[1].0 > w[0].1)
            .max_by_key(|w| w[1].0 - w[0].1)
            .map(|w| (w[0].1 + w[1].0) / 2)
            .unwrap_or(limit);
        chunks.push((start, cut));
        start = cut;
    }
    chunks.push((start, total_ms));
    chunks
}

/// Maps new speaker ids onto live ones by spoken overlap (greedy, largest first). Unmatched new
/// ids get ids no live speaker uses.
pub fn map_to_live(spans: &[Span], live: &[AttributedWord]) -> BTreeMap<u8, u8> {
    let mut overlap: BTreeMap<(u8, u8), u64> = BTreeMap::new();
    for w in live {
        let Some(l) = w.speaker else { continue };
        for s in spans {
            let o = w
                .word
                .t1_ms
                .min(s.t1_ms)
                .saturating_sub(w.word.t0_ms.max(s.t0_ms));
            if o > 0 {
                *overlap.entry((s.slot, l)).or_default() += o;
            }
        }
    }
    let mut pairs: Vec<((u8, u8), u64)> = overlap.into_iter().collect();
    pairs.sort_by_key(|a| std::cmp::Reverse(a.1));
    let mut map = BTreeMap::new();
    let mut used = std::collections::BTreeSet::new();
    for ((new, l), _) in pairs {
        if !map.contains_key(&new) && !used.contains(&l) {
            map.insert(new, l);
            used.insert(l);
        }
    }
    let mut next = 0u8;
    let mut new_ids: Vec<u8> = spans.iter().map(|s| s.slot).collect();
    new_ids.sort_unstable();
    new_ids.dedup();
    for id in new_ids {
        if map.contains_key(&id) {
            continue;
        }
        while used.contains(&next) {
            next += 1;
        }
        map.insert(id, next);
        used.insert(next);
    }
    map
}

/// Speaker parts of `[t0, t1)` according to `spans` (merged, in order).
fn parts(spans: &[Span], t0: u64, t1: u64) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for s in spans.iter().filter(|s| s.t1_ms > t0 && s.t0_ms < t1) {
        let (a, b) = (s.t0_ms.max(t0) - t0, s.t1_ms.min(t1) - t0);
        match runs.last_mut() {
            Some(r) if r.speaker == Some(s.slot) => r.t1_ms = b,
            _ => runs.push(Run {
                speaker: Some(s.slot),
                t0_ms: a,
                t1_ms: b,
                text: String::new(),
            }),
        }
    }
    runs
}

pub struct Refined {
    pub words: Vec<TimedWord>,
    pub activity: SpeakerActivity,
    /// How many live spans were split because they held several speakers.
    pub splits: usize,
    pub speakers: usize,
}

/// The full speaker pass. `live` = the finished live transcript's words; `utterances` = utterance
/// spans (ms) used to choose chunk cuts.
pub fn run(
    client: &Client,
    oa: &OpenAiSettings,
    wav_path: &Path,
    utterances: &[(u64, u64)],
    live: &[AttributedWord],
    languages: &[String],
    keywords: &[String],
) -> Result<Refined> {
    let wav = Wav::open(wav_path)?;
    let chunks = plan_chunks(wav.len_ms(), utterances, CHUNK_MS);
    let language = (languages.len() == 1).then(|| languages[0].clone());
    log::info!(
        "Final speaker pass: {:.1} min in {} chunk(s)",
        wav.len_ms() as f64 / 60_000.0,
        chunks.len()
    );

    // first chunk without references; its voices become the references for the rest
    let (c0a, c0b) = chunks[0];
    let pcm0 = wav.read(c0a, c0b)?;
    let segs0 = client.diarize(&oa.diarize_model, &pcm0, language.as_deref(), &[])?;
    let mut ids: BTreeMap<String, u8> = BTreeMap::new();
    let mut spans: Vec<Span> = Vec::new();
    let add = |ids: &mut BTreeMap<String, u8>, spans: &mut Vec<Span>, off: u64, s: &DiarSegment| {
        if s.end <= s.start {
            return;
        }
        let n = ids.len() as u8;
        let slot = *ids.entry(s.speaker.clone()).or_insert(n);
        spans.push(Span {
            t0_ms: off + (s.start * 1000.0) as u64,
            t1_ms: off + (s.end * 1000.0) as u64,
            slot,
        });
    };
    for s in &segs0 {
        add(&mut ids, &mut spans, c0a, s);
    }

    if chunks.len() > 1 {
        // references: the four voices that spoke most, ≤ 9.5 s each
        let mut talk: BTreeMap<&str, f64> = BTreeMap::new();
        for s in &segs0 {
            *talk.entry(s.speaker.as_str()).or_default() += s.end - s.start;
        }
        let mut by_talk: Vec<(&str, f64)> = talk.into_iter().collect();
        by_talk.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut known: Vec<(KnownSpeaker, String)> = Vec::new();
        for (label, _) in by_talk.into_iter().take(4) {
            let mut clip = Vec::new();
            for s in segs0.iter().filter(|s| s.speaker == label) {
                let a = ((s.start * 16_000.0) as usize).min(pcm0.len());
                let b = ((s.end * 16_000.0) as usize).min(pcm0.len());
                clip.extend_from_slice(&pcm0[a..b]);
                if clip.len() >= 152_000 {
                    break;
                }
            }
            clip.truncate(152_000);
            if clip.len() >= 32_000 {
                known.push((
                    KnownSpeaker {
                        name: format!("V{}", known.len() + 1),
                        data_url: wav_data_url(&clip),
                    },
                    label.to_string(),
                ));
            }
        }
        let refs: Vec<KnownSpeaker> = known.iter().map(|(k, _)| k.clone()).collect();
        let results: Vec<Result<Vec<DiarSegment>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunks[1..]
                .iter()
                .map(|&(a, b)| {
                    let (wav, refs, language) = (&wav, &refs, &language);
                    scope.spawn(move || {
                        let pcm = wav.read(a, b)?;
                        client.diarize(&oa.diarize_model, &pcm, language.as_deref(), refs)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("panic"))))
                .collect()
        });
        for (i, (res, &(a, _))) in results.into_iter().zip(&chunks[1..]).enumerate() {
            let segs = res.with_context(|| format!("chunk {}", i + 2))?;
            // known voice names map back to the first chunk's labels; other letters are
            // chunk-local
            for s in segs {
                let label = match known.iter().find(|(k, _)| k.name == s.speaker) {
                    Some((_, l)) => l.clone(),
                    None => format!("{}#{}", s.speaker, i + 2),
                };
                add(
                    &mut ids,
                    &mut spans,
                    a,
                    &DiarSegment {
                        speaker: label,
                        ..s
                    },
                );
            }
        }
    }

    let map = map_to_live(&spans, live);
    for s in &mut spans {
        s.slot = map[&s.slot];
    }
    spans.retain(|s| (s.slot as usize) < SLOTS);
    spans.sort_by_key(|s| s.t0_ms);

    // live spans holding several speakers: re-transcribe each part
    let mut words = Vec::new();
    let mut splits = 0;
    let mut to_split: Vec<(&TimedWord, Vec<Run>)> = Vec::new();
    for w in live.iter().map(|w| &w.word) {
        let p = parts(&spans, w.t0_ms, w.t1_ms);
        let long = p
            .iter()
            .filter(|r| r.t1_ms - r.t0_ms >= MIN_PART_MS)
            .count();
        if long >= 2 {
            to_split.push((
                w,
                p.into_iter()
                    .filter(|r| r.t1_ms - r.t0_ms >= MIN_PART_MS)
                    .collect(),
            ));
        } else {
            words.push(w.clone());
        }
    }
    for (w, runs) in to_split {
        let len = w.t1_ms - w.t0_ms;
        let bounds = slice_bounds(&runs, len);
        let texts: Vec<Option<String>> = std::thread::scope(|scope| {
            let hs: Vec<_> = bounds
                .iter()
                .map(|&(b0, b1)| {
                    let wav = &wav;
                    scope.spawn(move || {
                        let pcm = wav.read(w.t0_ms + b0, w.t0_ms + b1).ok()?;
                        client
                            .transcribe(&oa.transcribe_model, &pcm, languages, keywords)
                            .ok()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().ok().flatten()).collect()
        });
        if texts
            .iter()
            .all(|t| t.as_deref().is_some_and(|t| !t.trim().is_empty()))
        {
            splits += 1;
            for (&(b0, b1), t) in bounds.iter().zip(texts) {
                words.push(TimedWord::new(
                    w.t0_ms + b0,
                    w.t0_ms + b1,
                    t.unwrap_or_default().trim(),
                ));
            }
        } else {
            words.push(w.clone());
        }
    }

    let end_ms = wav
        .len_ms()
        .max(words.iter().map(|w| w.t1_ms).max().unwrap_or(0));
    let n_frames = (end_ms / FRAME_MS + 1) as usize;
    let mut probs = vec![0.0f32; n_frames * SLOTS];
    for s in &spans {
        for f in (s.t0_ms / FRAME_MS) as usize..((s.t1_ms / FRAME_MS) as usize).min(n_frames) {
            probs[f * SLOTS + s.slot as usize] = 1.0;
        }
    }
    let mut activity = SpeakerActivity::new(SLOTS, FRAME_MS);
    activity.write(0, &probs);
    activity.finalize_until(n_frames as u64);
    let speakers = {
        let mut v: Vec<u8> = spans.iter().map(|s| s.slot).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    Ok(Refined {
        words,
        activity,
        splits,
        speakers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_meetings_are_one_chunk() {
        assert_eq!(plan_chunks(90_000, &[], CHUNK_MS), vec![(0, 90_000)]);
    }

    #[test]
    fn long_meetings_are_cut_in_the_longest_late_pause() {
        let utts = [
            (0, 400_000),
            (410_000, 500_000),
            (530_000, 590_000),
            (591_000, 900_000),
        ];
        // pauses in the last third (≥ 400 s): 400–410 (10 s), 500–530 (30 s), 590–591
        assert_eq!(
            plan_chunks(900_000, &utts, 600_000),
            vec![(0, 515_000), (515_000, 900_000)]
        );
        // no pause at all: hard cuts
        assert_eq!(
            plan_chunks(1_300_000, &[], 600_000),
            vec![(0, 600_000), (600_000, 1_200_000), (1_200_000, 1_300_000)]
        );
    }

    fn aw(t0: u64, t1: u64, speaker: Option<u8>) -> AttributedWord {
        AttributedWord {
            word: TimedWord::new(t0, t1, "x"),
            speaker,
            confidence: 1.0,
            tentative: false,
        }
    }

    #[test]
    fn new_ids_follow_live_ids_by_overlap() {
        let spans = [
            Span {
                t0_ms: 0,
                t1_ms: 5_000,
                slot: 0,
            },
            Span {
                t0_ms: 5_000,
                t1_ms: 9_000,
                slot: 1,
            },
            Span {
                t0_ms: 9_000,
                t1_ms: 12_000,
                slot: 2,
            },
        ];
        // live called the first stretch 3 and merged the last two into 1
        let live = [aw(0, 5_000, Some(3)), aw(5_000, 12_000, Some(1))];
        let m = map_to_live(&spans, &live);
        assert_eq!(m[&0], 3);
        assert_eq!(m[&1], 1);
        // slot 2 lost the overlap contest and gets an unused id
        assert_eq!(m[&2], 0);
    }

    #[test]
    fn parts_merge_same_speaker_and_clip_to_the_span() {
        let spans = [
            Span {
                t0_ms: 0,
                t1_ms: 2_000,
                slot: 0,
            },
            Span {
                t0_ms: 2_100,
                t1_ms: 3_000,
                slot: 0,
            },
            Span {
                t0_ms: 3_200,
                t1_ms: 8_000,
                slot: 1,
            },
        ];
        let p = parts(&spans, 1_000, 6_000);
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].t0_ms, p[0].t1_ms, p[0].speaker), (0, 2_000, Some(0)));
        assert_eq!(
            (p[1].t0_ms, p[1].t1_ms, p[1].speaker),
            (2_200, 5_000, Some(1))
        );
    }
}
