//! Streaming speaker diarization for MeetScribe, backed by NVIDIA NeMo-Speech.cpp running
//! Nemotron 3 Diarization (GGUF, ggml with Metal on Apple Silicon).
//!
//! [`NativeDiarizer`] implements [`meeting_core::StreamingDiarizer`]: feed 16 kHz mono samples,
//! get back newly labelled 10 ms frames with per-speaker probabilities. Built without the SDK
//! (see `build.rs`), constructing it fails with a clear error instead.

mod ffi;

use std::path::Path;

use anyhow::Result;
use meeting_core::{DiarUpdate, StreamingDiarizer};

pub use ffi::nemo_speech_diar_segment as Segment;

/// Model geometry. `preset` is NeMo-Speech's name (`"v3-streaming"`, `"v3-offline"`; `None` =
/// low-latency default); the frame overrides are in 80 ms encoder frames (`0` = keep preset).
#[derive(Debug, Clone, Default)]
pub struct NativeConfig {
    pub gpu: Option<i32>,
    pub preset: Option<String>,
    pub chunk_frames: i32,
    pub right_context_frames: i32,
}

impl NativeConfig {
    /// `(chunk, right context)` in 80 ms frames for a MeetScribe latency profile name.
    pub fn for_profile(profile: &str) -> Self {
        let (chunk, rc) = match profile {
            "ultra_low_latency" => (3, 1),
            "very_low_latency" => (6, 2),
            "p5s" => (56, 8),
            _ => (9, 4),
        };
        Self {
            gpu: Some(0),
            preset: None,
            chunk_frames: chunk,
            right_context_frames: rc,
        }
    }

    /// Input buffer before a chunk is labelled.
    pub fn latency_ms(&self) -> u64 {
        ((self.chunk_frames.max(1) + self.right_context_frames.max(0)) * 80) as u64
    }
}

/// True when this build linked the NeMo-Speech.cpp SDK.
pub const AVAILABLE: bool = cfg!(nemo_speech);

#[cfg(nemo_speech)]
mod native {
    use super::*;
    use crate::ffi::*;
    use anyhow::{anyhow, bail};
    use std::ffi::{CStr, CString};

    fn last_error() -> String {
        unsafe {
            let p = nemo_speech_asr_last_error();
            if p.is_null() {
                "unknown error".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    fn check(status: nemo_speech_asr_status, what: &str) -> Result<()> {
        if status == NEMO_SPEECH_ASR_OK {
            Ok(())
        } else {
            Err(anyhow!("{what} failed ({status}): {}", last_error()))
        }
    }

    pub struct NativeDiarizer {
        model: *mut nemo_speech_diar_model,
        stream: *mut nemo_speech_diar_stream,
        num_speakers: usize,
        frame_ms: u64,
        latency_ms: u64,
        /// Frames already handed out.
        emitted: i64,
        finished: bool,
        describe: String,
        scratch: Vec<f32>,
    }

    // The handles are only used from the thread that owns the diarizer (streams are
    // single-threaded by contract); moving it between threads is fine.
    unsafe impl Send for NativeDiarizer {}

    impl NativeDiarizer {
        pub fn new(model_path: &Path, cfg: &NativeConfig) -> Result<Self> {
            let path = CString::new(model_path.to_string_lossy().as_bytes())?;
            let preset = cfg.preset.as_deref().map(CString::new).transpose()?;
            let c = nemo_speech_diar_model_config {
                size: std::mem::size_of::<nemo_speech_diar_model_config>(),
                model_path: path.as_ptr(),
                gpu: cfg.gpu.unwrap_or(-1),
                preset: preset.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
                chunk_frames: cfg.chunk_frames,
                right_context_frames: cfg.right_context_frames,
                left_context_frames: -1,
                fifo_frames: 0,
                spkcache_frames: 0,
                update_period_frames: 0,
            };
            let mut model = std::ptr::null_mut();
            check(
                unsafe { nemo_speech_diar_create(&c, &mut model) },
                "diar_create",
            )?;
            let mut stream = std::ptr::null_mut();
            if let Err(e) = check(
                unsafe { nemo_speech_diar_stream_open(model, &mut stream) },
                "stream_open",
            ) {
                unsafe { nemo_speech_diar_destroy(model) };
                return Err(e);
            }
            let num_speakers = unsafe { nemo_speech_diar_num_speakers(model) }.max(1) as usize;
            let frame_ms =
                (unsafe { nemo_speech_diar_seconds_per_frame(model) } * 1000.0).round() as u64;
            Ok(Self {
                model,
                stream,
                num_speakers,
                frame_ms: frame_ms.max(1),
                latency_ms: cfg.latency_ms(),
                emitted: 0,
                finished: false,
                describe: format!(
                    "{} (NeMo-Speech.cpp, chunk {}+{} x 80 ms, {})",
                    model_path.file_name().unwrap_or_default().to_string_lossy(),
                    cfg.chunk_frames,
                    cfg.right_context_frames,
                    if cfg.gpu.unwrap_or(-1) >= 0 {
                        "Metal"
                    } else {
                        "CPU"
                    }
                ),
                scratch: Vec::new(),
            })
        }

        /// Frames labelled since the last call. Labelled frames are final in Sortformer streaming.
        fn drain(&mut self) -> Result<DiarUpdate> {
            let count = unsafe { nemo_speech_diar_frame_count(self.stream) };
            let start = unsafe { nemo_speech_diar_frame_probs_start(self.stream) };
            if count <= self.emitted {
                return Ok(DiarUpdate {
                    start_frame: self.emitted as u64,
                    probs: Vec::new(),
                    finalized_until: self.emitted as u64,
                });
            }
            if start > self.emitted {
                bail!(
                    "diarizer compacted frames before they were read ({start} > {})",
                    self.emitted
                );
            }
            let retained = (count - start) as usize * self.num_speakers;
            self.scratch.resize(retained, 0.0);
            check(
                unsafe {
                    nemo_speech_diar_frame_probs(self.stream, self.scratch.as_mut_ptr(), retained)
                },
                "frame_probs",
            )?;
            let from = (self.emitted - start) as usize * self.num_speakers;
            let update = DiarUpdate {
                start_frame: self.emitted as u64,
                probs: self.scratch[from..].to_vec(),
                finalized_until: count as u64,
            };
            self.emitted = count;
            Ok(update)
        }

        /// Speaker segments with the library's default postprocessing (NeMo's callhome values).
        pub fn segments(&self) -> Result<Vec<Segment>> {
            let mut count = 0usize;
            check(
                unsafe {
                    nemo_speech_diar_segments(
                        self.stream,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        0,
                        &mut count,
                    )
                },
                "segments",
            )?;
            let mut out = vec![Segment::default(); count];
            check(
                unsafe {
                    nemo_speech_diar_segments(
                        self.stream,
                        std::ptr::null(),
                        out.as_mut_ptr(),
                        out.len(),
                        &mut count,
                    )
                },
                "segments",
            )?;
            out.truncate(count);
            Ok(out)
        }
    }

    impl StreamingDiarizer for NativeDiarizer {
        fn num_speakers(&self) -> usize {
            self.num_speakers
        }
        fn frame_ms(&self) -> u64 {
            self.frame_ms
        }
        fn latency_ms(&self) -> u64 {
            self.latency_ms
        }
        fn push(&mut self, samples: &[f32]) -> Result<DiarUpdate> {
            if !samples.is_empty() && !self.finished {
                check(
                    unsafe {
                        nemo_speech_diar_stream_push_f32(
                            self.stream,
                            samples.as_ptr(),
                            samples.len(),
                            16_000,
                        )
                    },
                    "stream_push",
                )?;
            }
            self.drain()
        }
        fn finish(&mut self) -> Result<DiarUpdate> {
            if !self.finished {
                self.finished = true;
                check(
                    unsafe { nemo_speech_diar_stream_finish(self.stream) },
                    "stream_finish",
                )?;
            }
            self.drain()
        }
        fn describe(&self) -> String {
            self.describe.clone()
        }
    }

    impl Drop for NativeDiarizer {
        fn drop(&mut self) {
            unsafe {
                nemo_speech_diar_stream_close(self.stream);
                nemo_speech_diar_destroy(self.model);
            }
        }
    }
}

#[cfg(nemo_speech)]
pub use native::NativeDiarizer;

#[cfg(not(nemo_speech))]
pub struct NativeDiarizer(());

#[cfg(not(nemo_speech))]
impl NativeDiarizer {
    pub fn new(_model_path: &Path, _cfg: &NativeConfig) -> Result<Self> {
        anyhow::bail!(
            "this build has no native diarizer: NeMo-Speech.cpp SDK was not found at build time \
             (run scripts/meetscribe/build_nemo_speech.sh and rebuild)"
        )
    }

    pub fn segments(&self) -> Result<Vec<Segment>> {
        unreachable!()
    }
}

#[cfg(not(nemo_speech))]
impl StreamingDiarizer for NativeDiarizer {
    fn num_speakers(&self) -> usize {
        unreachable!()
    }
    fn frame_ms(&self) -> u64 {
        unreachable!()
    }
    fn latency_ms(&self) -> u64 {
        unreachable!()
    }
    fn push(&mut self, _: &[f32]) -> Result<DiarUpdate> {
        unreachable!()
    }
    fn finish(&mut self) -> Result<DiarUpdate> {
        unreachable!()
    }
    fn describe(&self) -> String {
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_geometry_matches_the_spike() {
        assert_eq!(NativeConfig::for_profile("low_latency").latency_ms(), 1_040);
        assert_eq!(
            NativeConfig::for_profile("ultra_low_latency").latency_ms(),
            320
        );
        assert_eq!(NativeConfig::for_profile("p5s").latency_ms(), 5_120);
    }

    #[test]
    fn ffi_struct_layout_matches_c() {
        // size_t + ptr + int32 (+pad) + ptr + 6 x int32 on LP64
        assert_eq!(
            std::mem::size_of::<ffi::nemo_speech_diar_model_config>(),
            56
        );
        assert_eq!(
            std::mem::size_of::<ffi::nemo_speech_diar_segmentation_config>(),
            48
        );
        assert_eq!(std::mem::size_of::<Segment>(), 24);
    }

    #[test]
    fn missing_sdk_or_model_is_an_error_not_a_crash() {
        let r = NativeDiarizer::new(
            Path::new("/nonexistent/model.gguf"),
            &NativeConfig::default(),
        );
        assert!(r.is_err());
    }
}
