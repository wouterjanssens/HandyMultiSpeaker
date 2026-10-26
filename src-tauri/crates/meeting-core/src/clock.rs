//! The meeting's single source of time.
//!
//! Every 16 kHz sample delivered by the capture sink advances the clock by one, *before* VAD, so
//! the diarizer (which sees every sample) and the ASR worker (which sees VAD-cut utterances) share
//! one timeline. Utterances carry their absolute start sample; ASR word times (relative to the
//! utterance PCM) are shifted by it. Samples lost to capture overruns are reported separately via
//! [`SampleClock::note_dropped`]; the clock then inserts the gap so later audio keeps its wall
//! position.

/// Sample rate of the meeting timeline.
pub const SAMPLE_RATE: u64 = 16_000;

/// Monotonic sample counter for one meeting.
#[derive(Debug, Default, Clone)]
pub struct SampleClock {
    samples: u64,
    dropped: u64,
}

impl SampleClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances the clock by `n` delivered samples and returns the absolute index of the first one.
    pub fn advance(&mut self, n: usize) -> u64 {
        let start = self.samples;
        self.samples += n as u64;
        start
    }

    /// Records `n` samples (at 16 kHz) that never reached the sink; the timeline skips over them.
    pub fn note_dropped(&mut self, n: u64) {
        self.samples += n;
        self.dropped += n;
    }

    /// Samples elapsed since the meeting started, including dropped ones.
    pub fn now(&self) -> u64 {
        self.samples
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn now_ms(&self) -> u64 {
        samples_to_ms(self.samples)
    }
}

pub fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / SAMPLE_RATE
}

pub fn ms_to_samples(ms: u64) -> u64 {
    ms * SAMPLE_RATE / 1000
}

/// Shifts a time relative to an utterance's PCM onto the meeting timeline.
///
/// `rel_ms` may be slightly negative or exceed the utterance for some engines; it is clamped at 0.
pub fn utterance_to_meeting_ms(utterance_start_sample: u64, rel_ms: i64) -> u64 {
    let base = samples_to_ms(utterance_start_sample) as i64;
    (base + rel_ms).max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_returns_start_of_block() {
        let mut c = SampleClock::new();
        assert_eq!(c.advance(512), 0);
        assert_eq!(c.advance(512), 512);
        assert_eq!(c.now(), 1024);
    }

    #[test]
    fn dropped_samples_keep_wall_position() {
        let mut c = SampleClock::new();
        c.advance(16_000);
        c.note_dropped(1_600);
        assert_eq!(c.advance(10), 17_600);
        assert_eq!(c.dropped(), 1_600);
        assert_eq!(c.now_ms(), 1_100);
    }

    #[test]
    fn utterance_offset() {
        // utterance starts at 2.5 s, word at +120 ms
        assert_eq!(utterance_to_meeting_ms(40_000, 120), 2_620);
        assert_eq!(utterance_to_meeting_ms(0, -30), 0);
    }
}
