//! MeetScribe meeting-mode core.
//!
//! Pure logic shared by the live meeting pipeline, kept free of Tauri, audio and model code so it
//! can be unit-tested with synthetic timelines:
//!
//! - [`clock`]: the single monotonic 16 kHz sample clock both workers are stamped with.
//! - [`activity`]: per-frame speaker probabilities from the diarizer, with a finalization cursor.
//! - [`align`]: assigns a speaker to each ASR word (or segment) by summed activity over its span.
//! - [`turns`]: groups attributed words into turns, tentative until the diarizer finalizes them.
//! - [`speakers`]: maps arrival-ordered diarizer indices to display labels and user names.
//! - [`segmenter`]: cuts the continuous stream into utterances for the ASR worker.
//! - [`diarizer`]: the trait streaming diarizer backends implement.
//! - [`export`]: Markdown transcript and word-level JSON sidecar.

pub mod activity;
pub mod align;
pub mod clock;
pub mod diarizer;
pub mod export;
pub mod segmenter;
pub mod speakers;
pub mod turns;

pub use activity::SpeakerActivity;
pub use align::{AlignConfig, AttributedWord, TimedWord};
pub use clock::SampleClock;
pub use diarizer::{DiarUpdate, StreamingDiarizer};
pub use segmenter::{Segmenter, SegmenterConfig, Utterance};
pub use speakers::SpeakerMap;
pub use turns::{Turn, TurnBuilder, TurnEvent};
