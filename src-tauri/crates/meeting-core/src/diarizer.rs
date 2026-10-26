//! The interface every streaming diarizer backend implements.
//!
//! Backends: the native NeMo-Speech.cpp binding (target) and a temporary Python sidecar running
//! the Hugging Face reference implementation (prototype). The meeting pipeline only sees this
//! trait, so backends can be swapped without touching alignment or UI code.

/// Frames produced by one push: frame-major probabilities for frames
/// `[start_frame, start_frame + probs.len() / num_speakers)` on the meeting timeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiarUpdate {
    pub start_frame: u64,
    pub probs: Vec<f32>,
    /// Frames before this index will not be revised again.
    pub finalized_until: u64,
}

pub trait StreamingDiarizer: Send {
    /// Model capacity (8 for Nemotron 3 Diarization).
    fn num_speakers(&self) -> usize;
    /// Output cadence in milliseconds (10 for Nemotron 3 Diarization).
    fn frame_ms(&self) -> u64;
    /// Algorithmic latency: how far behind the input the finalized frames trail.
    fn latency_ms(&self) -> u64;
    /// Feeds 16 kHz mono samples that directly follow the previous push.
    fn push(&mut self, samples: &[f32]) -> anyhow::Result<DiarUpdate>;
    /// No more audio: labels the tail and finalizes everything.
    fn finish(&mut self) -> anyhow::Result<DiarUpdate>;
    /// Short human-readable backend/model description for logs and output metadata.
    fn describe(&self) -> String;
}
