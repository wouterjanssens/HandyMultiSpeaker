//! Live turn building.
//!
//! Words arrive from the ASR worker in utterance batches; speaker activity arrives from the
//! diarizer continuously and may be revised until finalized. [`TurnBuilder::refresh`] re-attributes
//! every not-yet-locked word against the latest activity, regroups those words into turns, locks
//! the leading turns that can no longer change, and returns the minimal list of [`TurnEvent`]s
//! the UI needs to mirror the transcript.
//!
//! Turn ids are positions in the transcript. Locked turns form a prefix and never change, so their
//! ids are stable; turns after them may be updated, appended or removed as attribution settles.

use serde::{Deserialize, Serialize};

use crate::activity::SpeakerActivity;
use crate::align::{attribute, AlignConfig, AttributedWord, TimedWord};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub id: u32,
    /// Arrival-ordered diarizer index; `None` when no speaker could be attributed.
    pub speaker: Option<u8>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Locked turns are final; others may still change speaker, split or merge.
    pub locked: bool,
    pub words: Vec<AttributedWord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnEvent {
    Appended { turn: Turn },
    Updated { turn: Turn },
    Removed { id: u32 },
}

#[derive(Debug, Clone, Copy)]
pub struct TurnConfig {
    pub align: AlignConfig,
    /// A pause longer than this starts a new turn even for the same speaker (readability).
    pub max_pause_ms: u64,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            align: AlignConfig::default(),
            max_pause_ms: 8_000,
        }
    }
}

#[derive(Debug, Default)]
pub struct TurnBuilder {
    cfg: TurnConfig,
    locked: Vec<Turn>,
    /// Words not yet part of a locked turn, sorted by start time.
    open_words: Vec<TimedWord>,
    /// Open turns as last emitted, for diffing.
    emitted_open: Vec<Turn>,
}

impl TurnBuilder {
    pub fn new(cfg: TurnConfig) -> Self {
        Self {
            cfg,
            ..Default::default()
        }
    }

    /// Queues recognised words (absolute times). Call [`Self::refresh`] afterwards.
    pub fn add_words(&mut self, words: impl IntoIterator<Item = TimedWord>) {
        for w in words {
            if w.text.trim().is_empty() {
                continue;
            }
            let pos = self.open_words.partition_point(|x| x.t0_ms <= w.t0_ms);
            self.open_words.insert(pos, w);
        }
    }

    /// Re-attributes open words against `activity`, locks settled turns and returns UI events.
    pub fn refresh(&mut self, activity: &SpeakerActivity) -> Vec<TurnEvent> {
        self.rebuild(activity, false)
    }

    /// Ends the meeting: everything still open is attributed once more and locked.
    pub fn finish(&mut self, activity: &SpeakerActivity) -> Vec<TurnEvent> {
        self.rebuild(activity, true)
    }

    /// All turns, locked first.
    pub fn turns(&self) -> impl Iterator<Item = &Turn> + Clone {
        self.locked.iter().chain(self.emitted_open.iter())
    }

    pub fn locked_count(&self) -> usize {
        self.locked.len()
    }

    fn rebuild(&mut self, activity: &SpeakerActivity, finish: bool) -> Vec<TurnEvent> {
        let attributed: Vec<AttributedWord> = self
            .open_words
            .iter()
            .map(|w| {
                let mut a = attribute(w, activity, &self.cfg.align);
                if finish {
                    a.tentative = false;
                }
                a
            })
            .collect();
        let mut open = group(attributed, self.cfg.max_pause_ms);

        // Lock leading turns that are entirely final and followed by a final word of the next
        // turn (so they can neither grow nor merge). On finish, lock everything.
        let mut n_lock = 0;
        while n_lock < open.len() {
            let turn = &open[n_lock];
            let all_final = turn.words.iter().all(|w| !w.tentative);
            let next_final = open
                .get(n_lock + 1)
                .map(|t| !t.words[0].tentative)
                .unwrap_or(finish);
            if all_final && next_final {
                n_lock += 1;
            } else {
                break;
            }
        }
        let locked_words: usize = open[..n_lock].iter().map(|t| t.words.len()).sum();
        self.open_words.drain(..locked_words);

        let base = self.locked.len() as u32;
        for (i, t) in open.iter_mut().enumerate() {
            t.id = base + i as u32;
            t.locked = i < n_lock;
        }

        let mut events = Vec::new();
        let previous = std::mem::take(&mut self.emitted_open);
        for (i, turn) in open.iter().enumerate() {
            match previous.get(i) {
                None => events.push(TurnEvent::Appended { turn: turn.clone() }),
                Some(prev) if prev != turn => {
                    events.push(TurnEvent::Updated { turn: turn.clone() })
                }
                _ => {}
            }
        }
        for prev in previous.iter().skip(open.len()) {
            events.push(TurnEvent::Removed { id: prev.id });
        }

        let still_open = open.split_off(n_lock);
        self.locked.extend(open);
        self.emitted_open = still_open;
        events
    }
}

/// Groups consecutive same-speaker words into turns (ids/locks assigned by the caller).
fn group(words: Vec<AttributedWord>, max_pause_ms: u64) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    for w in words {
        let extend = turns
            .last()
            .map(|t| {
                t.speaker == w.speaker && w.word.t0_ms.saturating_sub(t.end_ms) <= max_pause_ms
            })
            .unwrap_or(false);
        if extend {
            let t = turns.last_mut().expect("checked");
            t.end_ms = t.end_ms.max(w.word.t1_ms);
            t.words.push(w);
        } else {
            turns.push(Turn {
                id: 0,
                speaker: w.speaker,
                start_ms: w.word.t0_ms,
                end_ms: w.word.t1_ms,
                text: String::new(),
                locked: false,
                words: vec![w],
            });
        }
    }
    for t in &mut turns {
        t.text = join_words(t.words.iter().map(|w| w.word.text.as_str()));
    }
    turns
}

/// Joins ASR word strings, avoiding spaces before punctuation.
pub fn join_words<'a>(words: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for w in words {
        let w = w.trim();
        if w.is_empty() {
            continue;
        }
        let glue = w.starts_with(['.', ',', '!', '?', ';', ':', ')']) || out.ends_with(['(', '-']);
        if !out.is_empty() && !glue {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 2;

    /// Activity with `spans` = (t0_ms, t1_ms, speaker), finalized up to `final_ms`.
    fn activity(end_ms: u64, final_ms: u64, spans: &[(u64, u64, usize)]) -> SpeakerActivity {
        let frames = (end_ms / 10) as usize;
        let mut probs = vec![0.0f32; frames * N];
        for &(t0, t1, s) in spans {
            for f in (t0 / 10) as usize..(t1 / 10) as usize {
                probs[f * N + s] = 0.9;
            }
        }
        let mut a = SpeakerActivity::new(N, 10);
        a.write(0, &probs);
        a.finalize_until(final_ms / 10);
        a
    }

    fn words(spec: &[(u64, u64, &str)]) -> Vec<TimedWord> {
        spec.iter()
            .map(|&(a, b, t)| TimedWord::new(a, b, t))
            .collect()
    }

    fn texts(b: &TurnBuilder) -> Vec<(Option<u8>, String, bool)> {
        b.turns()
            .map(|t| (t.speaker, t.text.clone(), t.locked))
            .collect()
    }

    #[test]
    fn groups_consecutive_speakers() {
        let a = activity(4_000, 4_000, &[(0, 2_000, 0), (2_000, 4_000, 1)]);
        let mut b = TurnBuilder::new(TurnConfig::default());
        b.add_words(words(&[
            (100, 400, "goeiemorgen"),
            (450, 700, "iedereen"),
            (700, 750, "."),
            (2_100, 2_400, "dag"),
            (2_500, 2_900, "Wouter"),
        ]));
        let ev = b.refresh(&a);
        assert_eq!(ev.len(), 2);
        assert_eq!(
            texts(&b),
            vec![
                (Some(0), "goeiemorgen iedereen.".to_string(), true),
                (Some(1), "dag Wouter".to_string(), false), // last turn may still grow
            ]
        );
    }

    #[test]
    fn tentative_turn_is_revised_then_locked() {
        // diarizer first thinks 2-3 s is speaker 0, then revises it to speaker 1
        let mut b = TurnBuilder::new(TurnConfig::default());
        b.add_words(words(&[(100, 500, "eerst"), (2_100, 2_500, "daarna")]));
        let early = activity(3_000, 1_000, &[(0, 3_000, 0)]);
        let ev = b.refresh(&early);
        assert_eq!(ev.len(), 1);
        assert_eq!(
            texts(&b),
            vec![(Some(0), "eerst daarna".to_string(), false)]
        );

        let revised = activity(
            4_000,
            4_000,
            &[(0, 1_000, 0), (2_000, 3_000, 1), (3_000, 4_000, 0)],
        );
        let ev = b.refresh(&revised);
        assert_eq!(
            ev,
            vec![
                TurnEvent::Updated {
                    turn: b.turns().next().unwrap().clone()
                },
                TurnEvent::Appended {
                    turn: b.turns().nth(1).unwrap().clone()
                },
            ]
        );
        assert_eq!(
            texts(&b),
            vec![
                (Some(0), "eerst".to_string(), true),
                (Some(1), "daarna".to_string(), false)
            ]
        );
        assert_eq!(b.locked_count(), 1);
    }

    #[test]
    fn merge_after_revision_removes_turn() {
        let mut b = TurnBuilder::new(TurnConfig::default());
        b.add_words(words(&[(100, 500, "een"), (1_100, 1_500, "twee")]));
        let split = activity(2_000, 0, &[(0, 1_000, 0), (1_000, 2_000, 1)]);
        b.refresh(&split);
        assert_eq!(b.turns().count(), 2);
        let merged = activity(2_000, 0, &[(0, 2_000, 0)]);
        let ev = b.refresh(&merged);
        assert!(ev.contains(&TurnEvent::Removed { id: 1 }));
        assert_eq!(texts(&b), vec![(Some(0), "een twee".to_string(), false)]);
    }

    #[test]
    fn locked_turns_never_change() {
        let mut b = TurnBuilder::new(TurnConfig::default());
        b.add_words(words(&[(100, 500, "a"), (1_100, 1_500, "b")]));
        b.refresh(&activity(2_000, 2_000, &[(0, 1_000, 0), (1_000, 2_000, 1)]));
        assert_eq!(b.locked_count(), 1);
        // later activity contradicts the past, but locked words were removed from re-attribution
        b.refresh(&activity(2_000, 2_000, &[(0, 2_000, 1)]));
        assert_eq!(b.turns().next().unwrap().speaker, Some(0));
    }

    #[test]
    fn long_pause_splits_same_speaker() {
        let a = activity(20_000, 20_000, &[(0, 20_000, 0)]);
        let mut b = TurnBuilder::new(TurnConfig::default());
        b.add_words(words(&[(100, 500, "voor"), (15_000, 15_400, "na")]));
        b.finish(&a);
        assert_eq!(b.turns().count(), 2);
        assert!(b.turns().all(|t| t.locked));
    }

    #[test]
    fn late_words_are_inserted_in_order() {
        let a = activity(3_000, 3_000, &[(0, 3_000, 0)]);
        let mut b = TurnBuilder::new(TurnConfig::default());
        b.add_words(words(&[(2_000, 2_300, "derde")]));
        b.add_words(words(&[(100, 300, "eerste"), (1_000, 1_200, "tweede")]));
        b.finish(&a);
        assert_eq!(texts(&b)[0].1, "eerste tweede derde");
    }

    #[test]
    fn join_handles_punctuation() {
        assert_eq!(
            join_words(["Hallo", ",", "hoe", "gaat", "'t", "?"]),
            "Hallo, hoe gaat 't?"
        );
        assert_eq!(join_words([" a ", "", "b"]), "a b");
    }
}
