//! Per-frame speaker activity as reported by the streaming diarizer.
//!
//! Frames are indexed on the meeting timeline (frame `i` covers `[i*frame_ms, (i+1)*frame_ms)`).
//! The diarizer may revise recent frames until it finalizes them, so [`SpeakerActivity::write`]
//! overwrites any overlapping range, and [`SpeakerActivity::finalized_until_ms`] tells consumers
//! which part of the timeline is settled.

/// Frame-major speaker probabilities with a finalization cursor.
#[derive(Debug, Clone)]
pub struct SpeakerActivity {
    num_speakers: usize,
    frame_ms: u64,
    /// Absolute index of the first retained frame (older frames may be trimmed).
    first_frame: u64,
    probs: Vec<f32>,
    finalized_until: u64,
}

impl SpeakerActivity {
    pub fn new(num_speakers: usize, frame_ms: u64) -> Self {
        assert!(num_speakers > 0 && frame_ms > 0);
        Self {
            num_speakers,
            frame_ms,
            first_frame: 0,
            probs: Vec::new(),
            finalized_until: 0,
        }
    }

    pub fn num_speakers(&self) -> usize {
        self.num_speakers
    }

    pub fn frame_ms(&self) -> u64 {
        self.frame_ms
    }

    /// One past the last frame written.
    pub fn end_frame(&self) -> u64 {
        self.first_frame + (self.probs.len() / self.num_speakers) as u64
    }

    pub fn end_ms(&self) -> u64 {
        self.end_frame() * self.frame_ms
    }

    /// Writes frames starting at absolute frame `start`, overwriting revised frames and extending
    /// the timeline. A gap between the current end and `start` is filled with silence.
    pub fn write(&mut self, start: u64, probs: &[f32]) {
        assert_eq!(
            probs.len() % self.num_speakers,
            0,
            "probs must be frame-major"
        );
        let start = start.max(self.first_frame);
        let end = self.end_frame();
        if start > end {
            let gap = ((start - end) as usize) * self.num_speakers;
            self.probs.resize(self.probs.len() + gap, 0.0);
        }
        let offset = ((start - self.first_frame) as usize) * self.num_speakers;
        let overlap = (self.probs.len() - offset).min(probs.len());
        self.probs[offset..offset + overlap].copy_from_slice(&probs[..overlap]);
        self.probs.extend_from_slice(&probs[overlap..]);
    }

    /// Marks frames before `frame` as final (monotonic).
    pub fn finalize_until(&mut self, frame: u64) {
        self.finalized_until = self.finalized_until.max(frame.min(self.end_frame()));
    }

    pub fn finalized_until_ms(&self) -> u64 {
        self.finalized_until * self.frame_ms
    }

    /// Drops frames before `frame` to bound memory; queries before it then see silence.
    pub fn trim_before(&mut self, frame: u64) {
        let frame = frame.min(self.end_frame());
        if frame <= self.first_frame {
            return;
        }
        let n = ((frame - self.first_frame) as usize) * self.num_speakers;
        self.probs.drain(..n);
        self.first_frame = frame;
    }

    /// Probability of `speaker` at absolute `frame` (0 outside the retained range).
    pub fn prob(&self, frame: u64, speaker: usize) -> f32 {
        if frame < self.first_frame || frame >= self.end_frame() || speaker >= self.num_speakers {
            return 0.0;
        }
        self.probs[((frame - self.first_frame) as usize) * self.num_speakers + speaker]
    }

    /// Per-speaker summed probability over `[t0_ms, t1_ms)` (at least one frame).
    pub fn sum_over(&self, t0_ms: u64, t1_ms: u64) -> Vec<f32> {
        let f0 = t0_ms / self.frame_ms;
        let f1 = (t1_ms.div_ceil(self.frame_ms)).max(f0 + 1);
        let mut sums = vec![0.0; self.num_speakers];
        for f in f0..f1 {
            for (s, sum) in sums.iter_mut().enumerate() {
                *sum += self.prob(f, s);
            }
        }
        sums
    }

    /// The speaker most active in the frame nearest to `t_ms` (searching outward up to
    /// `max_search_ms`) whose probability exceeds `threshold`.
    pub fn nearest_active(&self, t_ms: u64, threshold: f32, max_search_ms: u64) -> Option<usize> {
        let center = (t_ms / self.frame_ms) as i64;
        let radius = (max_search_ms / self.frame_ms) as i64;
        for d in 0..=radius {
            for f in [center - d, center + d] {
                if f < 0 {
                    continue;
                }
                let (best, p) = (0..self.num_speakers)
                    .map(|s| (s, self.prob(f as u64, s)))
                    .fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
                if p > threshold {
                    return Some(best);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(rows: &[[f32; 2]]) -> Vec<f32> {
        rows.iter().flatten().copied().collect()
    }

    #[test]
    fn write_extends_and_overwrites() {
        let mut a = SpeakerActivity::new(2, 10);
        a.write(0, &frames(&[[0.9, 0.0], [0.9, 0.0], [0.1, 0.1]]));
        assert_eq!(a.end_frame(), 3);
        // revise frame 2, add frame 3
        a.write(2, &frames(&[[0.0, 0.8], [0.0, 0.9]]));
        assert_eq!(a.end_frame(), 4);
        assert_eq!(a.prob(2, 1), 0.8);
        assert_eq!(a.prob(3, 1), 0.9);
        assert_eq!(a.prob(0, 0), 0.9);
    }

    #[test]
    fn write_with_gap_fills_silence() {
        let mut a = SpeakerActivity::new(2, 10);
        a.write(5, &frames(&[[1.0, 0.0]]));
        assert_eq!(a.end_frame(), 6);
        assert_eq!(a.prob(3, 0), 0.0);
        assert_eq!(a.prob(5, 0), 1.0);
    }

    #[test]
    fn finalize_is_monotonic_and_bounded() {
        let mut a = SpeakerActivity::new(2, 10);
        a.write(0, &frames(&[[0.0, 0.0]; 10]));
        a.finalize_until(6);
        a.finalize_until(3);
        assert_eq!(a.finalized_until_ms(), 60);
        a.finalize_until(100);
        assert_eq!(a.finalized_until_ms(), 100);
    }

    #[test]
    fn trim_keeps_absolute_indices() {
        let mut a = SpeakerActivity::new(2, 10);
        a.write(0, &frames(&[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]));
        a.trim_before(2);
        assert_eq!(a.prob(1, 0), 0.0);
        assert_eq!(a.prob(2, 0), 0.3);
        a.write(3, &frames(&[[0.4, 0.0]]));
        assert_eq!(a.prob(3, 0), 0.4);
    }

    #[test]
    fn sum_and_nearest() {
        let mut a = SpeakerActivity::new(2, 10);
        // 0-50 ms speaker 0, 50-100 ms silence, 100-150 ms speaker 1
        let mut rows = vec![[0.9, 0.0]; 5];
        rows.extend(vec![[0.0, 0.0]; 5]);
        rows.extend(vec![[0.0, 0.8]; 5]);
        a.write(0, &frames(&rows));
        let s = a.sum_over(0, 50);
        assert!((s[0] - 4.5).abs() < 1e-5 && s[1] == 0.0);
        assert_eq!(a.nearest_active(70, 0.5, 100), Some(0)); // 70ms: 30ms back to spk0, 30ms fwd to spk1 -> backward checked first
        assert_eq!(a.nearest_active(90, 0.5, 100), Some(1));
        assert_eq!(a.nearest_active(75, 0.5, 10), None);
    }
}
