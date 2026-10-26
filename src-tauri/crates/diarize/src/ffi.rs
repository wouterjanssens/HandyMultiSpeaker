//! Raw declarations for `nemo_speech/diar.h` (+ the shared status/error API from `asr.h`),
//! NeMo-Speech.cpp v0.2.0. Size-prefixed structs are append-only upstream, so these stay valid
//! against newer libraries.
#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_double, c_float};

pub type nemo_speech_asr_status = i32;
pub const NEMO_SPEECH_ASR_OK: nemo_speech_asr_status = 0;

#[repr(C)]
pub struct nemo_speech_diar_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct nemo_speech_diar_stream {
    _private: [u8; 0],
}

#[repr(C)]
pub struct nemo_speech_diar_model_config {
    pub size: usize,
    pub model_path: *const c_char,
    /// GPU device index; -1 = CPU.
    pub gpu: i32,
    /// "v3-streaming" / "v3-offline" (V3), NULL = model default (low latency).
    pub preset: *const c_char,
    pub chunk_frames: i32,
    pub right_context_frames: i32,
    pub left_context_frames: i32,
    pub fifo_frames: i32,
    pub spkcache_frames: i32,
    pub update_period_frames: i32,
}

#[repr(C)]
pub struct nemo_speech_diar_segmentation_config {
    pub size: usize,
    pub onset: c_float,
    pub offset: c_float,
    pub pad_onset_sec: c_double,
    pub pad_offset_sec: c_double,
    pub min_gap_sec: c_double,
    pub min_duration_sec: c_double,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct nemo_speech_diar_segment {
    pub start_time: c_double,
    pub end_time: c_double,
    /// 1-based.
    pub speaker: i32,
}

#[cfg(nemo_speech)]
extern "C" {
    pub fn nemo_speech_asr_last_error() -> *const c_char;

    pub fn nemo_speech_diar_create(
        cfg: *const nemo_speech_diar_model_config,
        out: *mut *mut nemo_speech_diar_model,
    ) -> nemo_speech_asr_status;
    pub fn nemo_speech_diar_destroy(model: *mut nemo_speech_diar_model);
    pub fn nemo_speech_diar_num_speakers(model: *const nemo_speech_diar_model) -> i32;
    pub fn nemo_speech_diar_seconds_per_frame(model: *const nemo_speech_diar_model) -> c_double;

    pub fn nemo_speech_diar_stream_open(
        model: *mut nemo_speech_diar_model,
        out: *mut *mut nemo_speech_diar_stream,
    ) -> nemo_speech_asr_status;
    pub fn nemo_speech_diar_stream_push_f32(
        stream: *mut nemo_speech_diar_stream,
        samples: *const c_float,
        n_samples: usize,
        sample_rate: i32,
    ) -> nemo_speech_asr_status;
    pub fn nemo_speech_diar_stream_finish(
        stream: *mut nemo_speech_diar_stream,
    ) -> nemo_speech_asr_status;
    pub fn nemo_speech_diar_stream_close(stream: *mut nemo_speech_diar_stream);

    pub fn nemo_speech_diar_frame_count(stream: *const nemo_speech_diar_stream) -> i64;
    pub fn nemo_speech_diar_frame_probs_start(stream: *const nemo_speech_diar_stream) -> i64;
    pub fn nemo_speech_diar_frame_probs(
        stream: *const nemo_speech_diar_stream,
        out: *mut c_float,
        capacity_floats: usize,
    ) -> nemo_speech_asr_status;
    pub fn nemo_speech_diar_segments(
        stream: *const nemo_speech_diar_stream,
        cfg: *const nemo_speech_diar_segmentation_config,
        out: *mut nemo_speech_diar_segment,
        capacity: usize,
        count: *mut usize,
    ) -> nemo_speech_asr_status;
}
