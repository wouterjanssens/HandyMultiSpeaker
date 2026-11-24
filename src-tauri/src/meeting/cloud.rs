//! meetscribe: meeting mode on OpenAI (no local ASR or diarizer model).
//!
//! The capture tap and the utterance segmenter stay local. Each utterance then goes through two
//! in-order stages on their own threads:
//!
//! 1. **text**: `gpt-transcribe` (language list + custom words). The text appears at once as one
//!    tentative span covering the utterance.
//! 2. **speakers**: `gpt-4o-transcribe-diarize` with the voices enrolled so far. The speaker runs
//!    become activity frames on the meeting timeline (so the regular aligner/turn builder does the
//!    rest), and the utterance is finalized. When one utterance holds several speakers, each run
//!    is re-transcribed with `gpt-transcribe` and replaces the single span.
//!
//! Stage 2 runs after stage 1 for the same utterance, so a span is never replaced before it was
//! added, and finalization only moves forward.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use meeting_core::clock::samples_to_ms;
use meeting_core::{DiarUpdate, StreamingDiarizer, TimedWord, Utterance};
use tauri::{AppHandle, Emitter};

use super::session::Shared;
use crate::openai::speakers::{slice_bounds, SpeakerTracker};
use crate::openai::{Client, OpenAiSettings};

/// Speaker slots on the activity timeline (only the first four can be enrolled voices).
const SLOTS: usize = 8;
const FRAME_MS: u64 = 10;
/// Runs shorter than this keep the diarizer's text instead of a second transcription.
const MIN_RETRANSCRIBE_MS: u64 = 700;

/// Stands in for the streaming diarizer: speaker activity is written by the speaker stage, so
/// this only lets the existing diarizer worker keep the clock (and the optional WAV) going.
pub struct CloudDiarizer {
    pub model: String,
}

impl StreamingDiarizer for CloudDiarizer {
    fn num_speakers(&self) -> usize {
        SLOTS
    }
    fn frame_ms(&self) -> u64 {
        FRAME_MS
    }
    fn latency_ms(&self) -> u64 {
        0
    }
    fn push(&mut self, _samples: &[f32]) -> anyhow::Result<DiarUpdate> {
        Ok(DiarUpdate::default())
    }
    fn finish(&mut self) -> anyhow::Result<DiarUpdate> {
        Ok(DiarUpdate::default())
    }
    fn describe(&self) -> String {
        format!("{} (OpenAI)", self.model)
    }
}

pub struct CloudConfig {
    pub client: Client,
    pub settings: OpenAiSettings,
    pub keywords: Vec<String>,
}

/// Languages for a request: the meeting's explicit choice, else the configured list.
fn languages(shared: &Arc<Mutex<Shared>>, settings: &OpenAiSettings) -> Vec<String> {
    let lang = shared.lock().unwrap().language.clone();
    if lang == "auto" {
        settings.languages.clone()
    } else {
        vec![lang]
    }
}

/// Runs both stages until `utterances` closes; returns when everything was processed.
pub fn run(
    app: AppHandle,
    shared: Arc<Mutex<Shared>>,
    cfg: CloudConfig,
    utterances: Receiver<Utterance>,
) {
    let cfg = Arc::new(cfg);
    let (tx, rx) = mpsc::channel::<(Utterance, Option<String>)>();
    let speakers = {
        let (app, shared, cfg) = (app.clone(), Arc::clone(&shared), Arc::clone(&cfg));
        std::thread::Builder::new()
            .name("meeting-openai-speakers".into())
            .spawn(move || speaker_stage(app, shared, cfg, rx))
            .expect("spawn speaker stage")
    };
    for u in utterances.iter() {
        let text = text_stage(&app, &shared, &cfg, &u);
        if tx.send((u, text)).is_err() {
            break;
        }
    }
    drop(tx);
    let _ = speakers.join();
}

fn text_stage(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    cfg: &CloudConfig,
    u: &Utterance,
) -> Option<String> {
    let start_ms = samples_to_ms(u.start_sample);
    let end_sample = u.start_sample + u.pcm.len() as u64;
    let end_ms = samples_to_ms(end_sample);
    let started = Instant::now();
    let langs = languages(shared, &cfg.settings);
    let text = match cfg.client.transcribe(
        &cfg.settings.transcribe_model,
        &u.pcm,
        &langs,
        &cfg.keywords,
    ) {
        Ok(t) => t.trim().to_string(),
        Err(e) => {
            log::warn!("OpenAI utterance transcription failed: {e:#}");
            let _ = app.emit("meeting:error", format!("{e:#}"));
            return None;
        }
    };
    let mut s = shared.lock().unwrap();
    s.stats.utterances += 1;
    let delay = s.audio_ms.saturating_sub(end_ms) as u32;
    s.stats.last_asr_delay_ms = delay;
    s.stats.max_asr_delay_ms = s.stats.max_asr_delay_ms.max(delay);
    s.push_asr_delay(delay);
    s.utterance_spans.push((start_ms, end_ms));
    log::debug!(
        "openai utterance {:.1}s → {} chars in {} ms",
        (end_ms - start_ms) as f32 / 1000.0,
        text.len(),
        started.elapsed().as_millis()
    );
    if text.is_empty() {
        return None;
    }
    s.turns
        .add_words([TimedWord::new(start_ms, end_ms.max(start_ms + 1), &text)]);
    s.refresh_and_emit(app);
    Some(text)
}

fn speaker_stage(
    app: AppHandle,
    shared: Arc<Mutex<Shared>>,
    cfg: Arc<CloudConfig>,
    rx: Receiver<(Utterance, Option<String>)>,
) {
    let mut tracker = SpeakerTracker::new();
    for (u, text) in rx.iter() {
        let start_ms = samples_to_ms(u.start_sample);
        let len_ms = samples_to_ms(u.pcm.len() as u64);
        let end_ms = start_ms + len_ms;
        let langs = languages(&shared, &cfg.settings);
        let lang = (langs.len() == 1).then(|| langs[0].clone());

        let runs = match cfg.client.diarize(
            &cfg.settings.diarize_model,
            &u.pcm,
            lang.as_deref(),
            &tracker.known(),
        ) {
            Ok(segments) => tracker.resolve(&segments, &u.pcm),
            Err(e) => {
                log::warn!("OpenAI diarization failed: {e:#}");
                Vec::new()
            }
        };
        let bounds = slice_bounds(&runs, len_ms);

        // several speakers in one utterance: re-transcribe each run for the best text
        let split =
            text.is_some() && runs.len() > 1 && runs.iter().any(|r| r.speaker != runs[0].speaker);
        let run_words: Option<Vec<TimedWord>> = split.then(|| {
            let texts: Vec<String> = std::thread::scope(|scope| {
                let handles: Vec<_> = runs
                    .iter()
                    .zip(&bounds)
                    .map(|(run, &(b0, b1))| {
                        let (cfg, pcm, langs) = (&cfg, &u.pcm, &langs);
                        scope.spawn(move || {
                            if b1 - b0 < MIN_RETRANSCRIBE_MS {
                                return run.text.trim().to_string();
                            }
                            let a = (b0 as usize * 16).min(pcm.len());
                            let b = (b1 as usize * 16).min(pcm.len());
                            cfg.client
                                .transcribe(
                                    &cfg.settings.transcribe_model,
                                    &pcm[a..b],
                                    langs,
                                    &cfg.keywords,
                                )
                                .map(|t| t.trim().to_string())
                                .unwrap_or_else(|e| {
                                    log::warn!("OpenAI run transcription failed: {e:#}");
                                    run.text.trim().to_string()
                                })
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_default())
                    .collect()
            });
            bounds
                .iter()
                .zip(texts)
                .filter(|(_, t)| !t.is_empty())
                .map(|(&(b0, b1), t)| {
                    TimedWord::new(start_ms + b0, (start_ms + b1).max(start_ms + b0 + 1), t)
                })
                .collect()
        });

        // speaker activity over the whole utterance, one slot per run
        let first_frame = start_ms / FRAME_MS;
        let n_frames = (len_ms / FRAME_MS) as usize;
        let mut probs = vec![0.0f32; n_frames * SLOTS];
        for (run, &(b0, b1)) in runs.iter().zip(&bounds) {
            let Some(slot) = run.speaker else { continue };
            let f0 = (b0 / FRAME_MS) as usize;
            let f1 = ((b1 / FRAME_MS) as usize).min(n_frames);
            for f in f0..f1 {
                probs[f * SLOTS + slot as usize] = 1.0;
            }
        }

        let mut s = shared.lock().unwrap();
        if let Some(words) = run_words {
            if !words.is_empty() {
                s.turns.replace_span(start_ms, end_ms, words);
            }
        }
        s.activity.write(first_frame, &probs);
        s.activity.finalize_until(first_frame + n_frames as u64);
        s.refresh_and_emit(&app);
    }
    log::info!(
        "OpenAI meeting speakers: {} voices enrolled",
        tracker.voice_count()
    );
}
