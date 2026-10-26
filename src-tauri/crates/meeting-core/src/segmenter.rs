//! Cuts the continuous meeting stream into utterances for the ASR worker.
//!
//! Fed fixed-size frames with a speech probability each (from any VAD), it emits an utterance
//! once speech is followed by enough silence, or forcibly when an utterance reaches its maximum
//! length (cutting at the least speech-like frame of the last few seconds, so long monologues
//! still appear on screen). Every utterance carries its absolute start sample on the meeting
//! clock, including the pre-roll it was given.

#[derive(Debug, Clone, Copy)]
pub struct SegmenterConfig {
    pub sample_rate: u32,
    /// Probability above which a frame counts as speech.
    pub threshold: f32,
    /// Speech needed to open an utterance.
    pub onset_ms: u32,
    /// Silence that closes an utterance.
    pub end_silence_ms: u32,
    /// Audio kept before the onset (catches soft word starts).
    pub pre_roll_ms: u32,
    /// Utterances are split when they reach this length.
    pub max_utterance_ms: u32,
    /// Window at the end of a too-long utterance searched for the best cut.
    pub split_search_ms: u32,
    /// Utterances with less speech than this are dropped (coughs, clicks).
    pub min_speech_ms: u32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            threshold: 0.4,
            onset_ms: 90,
            end_silence_ms: 600,
            pre_roll_ms: 300,
            max_utterance_ms: 25_000,
            split_search_ms: 4_000,
            min_speech_ms: 250,
        }
    }
}

/// A span of meeting audio to transcribe.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    /// Absolute index of `pcm[0]` on the meeting clock.
    pub start_sample: u64,
    pub pcm: Vec<f32>,
    /// True when cut because of the length cap rather than a pause.
    pub forced_split: bool,
}

#[derive(Debug)]
pub struct Segmenter {
    cfg: SegmenterConfig,
    frame_samples: usize,
    /// Pre-roll ring while idle: (start sample, frame).
    pre_roll: std::collections::VecDeque<(u64, Vec<f32>)>,
    /// Frames of the utterance being collected, with their probabilities.
    current: Vec<(u64, Vec<f32>, f32)>,
    in_speech: bool,
    onset_run: usize,
    silence_run: usize,
    speech_frames: usize,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig, frame_samples: usize) -> Self {
        assert!(frame_samples > 0);
        Self {
            cfg,
            frame_samples,
            pre_roll: Default::default(),
            current: Vec::new(),
            in_speech: false,
            onset_run: 0,
            silence_run: 0,
            speech_frames: 0,
        }
    }

    fn frames_for(&self, ms: u32) -> usize {
        ((ms as usize * self.cfg.sample_rate as usize) / 1000).div_ceil(self.frame_samples)
    }

    /// Feeds one frame starting at absolute sample `start`; returns a finished utterance, if any.
    pub fn push(&mut self, start: u64, frame: &[f32], prob: f32) -> Option<Utterance> {
        let speech = prob >= self.cfg.threshold;
        if !self.in_speech {
            self.pre_roll.push_back((start, frame.to_vec()));
            self.onset_run = if speech { self.onset_run + 1 } else { 0 };
            let keep = self.frames_for(self.cfg.pre_roll_ms) + self.onset_run;
            while self.pre_roll.len() > keep.max(1) {
                self.pre_roll.pop_front();
            }
            if self.onset_run >= self.frames_for(self.cfg.onset_ms).max(1) {
                self.in_speech = true;
                self.silence_run = 0;
                self.speech_frames = self.onset_run;
                // pre-roll frames count as non-speech for split scoring
                self.current = self.pre_roll.drain(..).map(|(s, f)| (s, f, 0.0)).collect();
                if let Some(last) = self.current.last_mut() {
                    last.2 = prob;
                }
            }
            return None;
        }

        self.current.push((start, frame.to_vec(), prob));
        if speech {
            self.speech_frames += 1;
            self.silence_run = 0;
        } else {
            self.silence_run += 1;
        }

        if self.silence_run >= self.frames_for(self.cfg.end_silence_ms) {
            // keep a little of the trailing silence, drop the rest
            let keep_tail = self.frames_for(200).min(self.silence_run);
            let drop = self.silence_run - keep_tail;
            self.current.truncate(self.current.len() - drop);
            return self.close(false);
        }
        if self.current.len() >= self.frames_for(self.cfg.max_utterance_ms) {
            return self.force_split();
        }
        None
    }

    /// Flushes whatever is being collected (end of meeting).
    pub fn flush(&mut self) -> Option<Utterance> {
        if self.in_speech {
            self.close(false)
        } else {
            None
        }
    }

    fn close(&mut self, forced: bool) -> Option<Utterance> {
        let frames = std::mem::take(&mut self.current);
        let speech_frames = self.speech_frames;
        self.in_speech = false;
        self.onset_run = 0;
        self.silence_run = 0;
        self.speech_frames = 0;
        self.pre_roll.clear();
        if speech_frames < self.frames_for(self.cfg.min_speech_ms) || frames.is_empty() {
            return None;
        }
        Some(Self::assemble(frames, forced))
    }

    fn force_split(&mut self) -> Option<Utterance> {
        let search = self.frames_for(self.cfg.split_search_ms).min(self.current.len() - 1);
        let from = self.current.len() - search;
        let cut = (from..self.current.len())
            .min_by(|&a, &b| self.current[a].2.total_cmp(&self.current[b].2))
            .unwrap_or(self.current.len() - 1)
            .max(1);
        let rest = self.current.split_off(cut);
        let head = std::mem::replace(&mut self.current, rest);
        self.speech_frames = self.current.iter().filter(|f| f.2 >= self.cfg.threshold).count();
        self.silence_run = self
            .current
            .iter()
            .rev()
            .take_while(|f| f.2 < self.cfg.threshold)
            .count();
        Some(Self::assemble(head, true))
    }

    fn assemble(frames: Vec<(u64, Vec<f32>, f32)>, forced: bool) -> Utterance {
        let start_sample = frames[0].0;
        let mut pcm = Vec::with_capacity(frames.iter().map(|f| f.1.len()).sum());
        for (_, f, _) in frames {
            pcm.extend_from_slice(&f);
        }
        Utterance {
            start_sample,
            pcm,
            forced_split: forced,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: usize = 160; // 10 ms frames keep the arithmetic readable

    /// Feeds `pattern` (one char per 10 ms frame: 's' speech, '.' silence).
    fn run(cfg: SegmenterConfig, pattern: &str) -> Vec<Utterance> {
        let mut seg = Segmenter::new(cfg, FS);
        let mut out = Vec::new();
        for (i, c) in pattern.chars().enumerate() {
            let start = (i * FS) as u64;
            let frame = vec![if c == 's' { 0.5 } else { 0.0 }; FS];
            let prob = if c == 's' { 0.9 } else { 0.05 };
            out.extend(seg.push(start, &frame, prob));
        }
        out.extend(seg.flush());
        out
    }

    fn cfg() -> SegmenterConfig {
        SegmenterConfig {
            onset_ms: 30,
            end_silence_ms: 100,
            pre_roll_ms: 50,
            max_utterance_ms: 1_000,
            split_search_ms: 300,
            min_speech_ms: 50,
            ..Default::default()
        }
    }

    #[test]
    fn single_utterance_with_pre_roll_and_tail() {
        let pattern = format!("{}{}{}", ".".repeat(20), "s".repeat(30), ".".repeat(20));
        let u = run(cfg(), &pattern);
        assert_eq!(u.len(), 1);
        // pre-roll of 5 frames before the onset frame run (which starts at frame 20)
        assert_eq!(u[0].start_sample, (15 * FS) as u64);
        // 5 pre-roll + 30 speech + 10 frames of silence trigger, of which 10 kept as tail (200ms cap)
        assert_eq!(u[0].pcm.len(), (5 + 30 + 10) * FS);
        assert!(!u[0].forced_split);
    }

    #[test]
    fn short_noise_is_ignored() {
        let pattern = format!("{}{}{}", ".".repeat(10), "ss", ".".repeat(20));
        assert!(run(cfg(), &pattern).is_empty());
        // 3 frames reach onset but stay under min speech (5 frames)
        let pattern = format!("{}{}{}", ".".repeat(10), "sss", ".".repeat(20));
        assert!(run(cfg(), &pattern).is_empty());
    }

    #[test]
    fn pause_splits_utterances() {
        let pattern = format!("{}{}{}{}", "s".repeat(20), ".".repeat(15), "s".repeat(20), ".".repeat(15));
        let u = run(cfg(), &pattern);
        assert_eq!(u.len(), 2);
        assert!(u[1].start_sample > u[0].start_sample + u[0].pcm.len() as u64 - 1);
    }

    #[test]
    fn long_monologue_is_force_split_at_a_dip() {
        // 2 s of speech with a short dip at 0.8 s (shorter than end_silence)
        let pattern = format!("{}{}{}", "s".repeat(80), "...", "s".repeat(117));
        let u = run(cfg(), &pattern);
        assert!(u.len() >= 2);
        assert!(u[0].forced_split);
        // cut lands in the dip (frames 80..83)
        let cut = (u[0].start_sample as usize + u[0].pcm.len()) / FS;
        assert!((80..=83).contains(&cut), "cut at frame {cut}");
        // utterances tile the audio without overlap or loss
        assert_eq!(u[1].start_sample as usize, u[0].start_sample as usize + u[0].pcm.len());
    }

    #[test]
    fn timeline_is_absolute() {
        let mut seg = Segmenter::new(cfg(), FS);
        let base = 1_000_000u64;
        let mut out = None;
        for i in 0..60 {
            let speech = (10..40).contains(&i);
            let r = seg.push(base + (i * FS) as u64, &[0.0; FS], if speech { 0.9 } else { 0.0 });
            out = out.or(r);
        }
        assert_eq!(out.unwrap().start_sample, base + (5 * FS) as u64);
    }
}
