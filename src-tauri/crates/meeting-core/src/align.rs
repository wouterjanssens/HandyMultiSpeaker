//! Word → speaker attribution.
//!
//! Each word (or a whole segment, for engines without word timing such as Whisper) gets the
//! speaker with the highest summed activity over its span. Words that fall in a diarizer gap take
//! the nearest active speaker within [`AlignConfig::max_gap_search_ms`]. A word is *tentative*
//! while any part of its span is past the diarizer's finalization cursor.

use serde::{Deserialize, Serialize};

use crate::activity::SpeakerActivity;

/// A recognised word on the meeting timeline (absolute milliseconds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedWord {
    pub t0_ms: u64,
    pub t1_ms: u64,
    pub text: String,
}

impl TimedWord {
    pub fn new(t0_ms: u64, t1_ms: u64, text: impl Into<String>) -> Self {
        Self {
            t0_ms,
            t1_ms: t1_ms.max(t0_ms),
            text: text.into(),
        }
    }
}

/// A word with its attributed speaker (arrival-ordered diarizer index, `None` = unknown).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributedWord {
    #[serde(flatten)]
    pub word: TimedWord,
    pub speaker: Option<u8>,
    /// Share of the span's total activity that went to `speaker` (0 when taken from a neighbour).
    pub confidence: f32,
    pub tentative: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct AlignConfig {
    /// Minimum summed activity per frame (averaged over the span) to trust the span itself.
    pub min_mean_activity: f32,
    /// Probability above which a frame counts as active when searching for the nearest speaker.
    pub nearest_threshold: f32,
    /// How far around a word in a diarizer gap to look for a speaker.
    pub max_gap_search_ms: u64,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            min_mean_activity: 0.15,
            nearest_threshold: 0.3,
            max_gap_search_ms: 2_000,
        }
    }
}

/// Attributes one word.
pub fn attribute(
    word: &TimedWord,
    activity: &SpeakerActivity,
    cfg: &AlignConfig,
) -> AttributedWord {
    let tentative = word.t1_ms > activity.finalized_until_ms();
    let sums = activity.sum_over(word.t0_ms, word.t1_ms);
    let frames = (word.t1_ms.saturating_sub(word.t0_ms) / activity.frame_ms()).max(1) as f32;
    let total: f32 = sums.iter().sum();
    let (best, best_sum) = sums
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });

    if best_sum / frames >= cfg.min_mean_activity {
        return AttributedWord {
            word: word.clone(),
            speaker: Some(best as u8),
            confidence: if total > 0.0 { best_sum / total } else { 0.0 },
            tentative,
        };
    }

    let mid = (word.t0_ms + word.t1_ms) / 2;
    let speaker = activity
        .nearest_active(mid, cfg.nearest_threshold, cfg.max_gap_search_ms)
        .map(|s| s as u8);
    // a gap word's neighbourhood may still change if it reaches into unfinalized frames
    let tentative = tentative || mid + cfg.max_gap_search_ms > activity.finalized_until_ms();
    AttributedWord {
        word: word.clone(),
        speaker,
        confidence: 0.0,
        tentative,
    }
}

/// Attributes a batch of words.
pub fn attribute_all(
    words: &[TimedWord],
    activity: &SpeakerActivity,
    cfg: &AlignConfig,
) -> Vec<AttributedWord> {
    words.iter().map(|w| attribute(w, activity, cfg)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 ms frames, 3 speakers; `spans` = (t0_ms, t1_ms, speaker, prob).
    fn activity(end_ms: u64, spans: &[(u64, u64, usize, f32)]) -> SpeakerActivity {
        let n = (end_ms / 10) as usize;
        let mut probs = vec![0.0f32; n * 3];
        for &(t0, t1, s, p) in spans {
            for f in (t0 / 10) as usize..(t1 / 10) as usize {
                probs[f * 3 + s] = p;
            }
        }
        let mut a = SpeakerActivity::new(3, 10);
        a.write(0, &probs);
        a.finalize_until(n as u64);
        a
    }

    #[test]
    fn word_takes_dominant_speaker() {
        let a = activity(2_000, &[(0, 1_000, 0, 0.9), (800, 2_000, 1, 0.9)]);
        let w = attribute(
            &TimedWord::new(850, 1_400, "hallo"),
            &a,
            &AlignConfig::default(),
        );
        assert_eq!(w.speaker, Some(1));
        assert!(w.confidence > 0.7);
        assert!(!w.tentative);
    }

    #[test]
    fn overlap_split_goes_to_larger_share() {
        // speaker 0 covers 60% of the word, speaker 1 40%
        let a = activity(1_000, &[(0, 600, 0, 0.9), (600, 1_000, 1, 0.9)]);
        let w = attribute(&TimedWord::new(0, 1_000, "ja"), &a, &AlignConfig::default());
        assert_eq!(w.speaker, Some(0));
    }

    #[test]
    fn gap_word_takes_nearest_speaker() {
        let a = activity(3_000, &[(0, 1_000, 2, 0.9), (2_500, 3_000, 0, 0.9)]);
        let w = attribute(
            &TimedWord::new(1_100, 1_300, "eh"),
            &a,
            &AlignConfig::default(),
        );
        assert_eq!(w.speaker, Some(2));
        assert_eq!(w.confidence, 0.0);
    }

    #[test]
    fn isolated_word_is_unknown() {
        let a = activity(10_000, &[(0, 500, 0, 0.9)]);
        let w = attribute(
            &TimedWord::new(8_000, 8_300, "oké"),
            &a,
            &AlignConfig::default(),
        );
        assert_eq!(w.speaker, None);
    }

    #[test]
    fn word_past_finalization_is_tentative() {
        let mut a = SpeakerActivity::new(3, 10);
        let mut probs = vec![0.0f32; 300];
        for f in 0..100 {
            probs[f * 3] = 0.9;
        }
        a.write(0, &probs);
        a.finalize_until(50); // 500 ms final
        let cfg = AlignConfig::default();
        assert!(!attribute(&TimedWord::new(100, 400, "een"), &a, &cfg).tentative);
        assert!(attribute(&TimedWord::new(400, 700, "twee"), &a, &cfg).tentative);
    }

    #[test]
    fn segment_level_engine_uses_whole_span() {
        // Whisper-style: one "word" per segment spanning both speakers; the majority wins
        let a = activity(5_000, &[(0, 1_000, 0, 0.9), (1_000, 5_000, 1, 0.8)]);
        let w = attribute(
            &TimedWord::new(0, 5_000, "a whole segment"),
            &a,
            &AlignConfig::default(),
        );
        assert_eq!(w.speaker, Some(1));
    }
}
