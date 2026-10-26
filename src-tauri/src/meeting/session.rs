//! One running meeting: capture tap → diarizer worker + ASR worker → shared transcript state.
//!
//! Timeline: the capture tap stamps every 16 kHz frame with its absolute sample index (one
//! counter, pre-VAD). The diarizer receives every sample from index 0, so its frame `i` covers
//! `[i*10ms, (i+1)*10ms)` of the same timeline. The ASR worker runs its own VAD/segmenter on the
//! same stamped frames and shifts each transcript's word times by the utterance's start sample.
//! Known gap: frames dropped by a capture-ring overrun are not visible per chunk, so such a gap
//! shifts everything after it (both workers equally); it is logged at stop.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use meeting_core::clock::{samples_to_ms, utterance_to_meeting_ms, SAMPLE_RATE};
use meeting_core::export::{self, MeetingInfo};
use meeting_core::turns::TurnConfig;
use meeting_core::{
    DiarUpdate, Segmenter, SegmenterConfig, SpeakerActivity, SpeakerMap, StreamingDiarizer,
    TimedWord, TurnBuilder, TurnEvent, Utterance,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use transcribe_cpp::Transcript;

use super::settings::MeetingSettings;
use crate::managers::audio::AudioRecordingManager;
use crate::managers::transcription::TranscriptionManager;

pub const MEETING_BINDING: &str = "toggle_meeting";

/// Absolute-stamped capture frame.
struct Frame {
    start: u64,
    samples: Vec<f32>,
}

/// Transcript state shared by both workers and the UI commands.
pub struct Shared {
    pub activity: SpeakerActivity,
    pub turns: TurnBuilder,
    pub speakers: SpeakerMap,
    max_speakers: usize,
    pub started_local: String,
    pub started_epoch_ms: u64,
    pub asr_model: String,
    pub diarizer: String,
    pub diarizer_latency_ms: u64,
    /// Meeting time covered by audio received so far.
    pub audio_ms: u64,
    /// Current language hint; changeable mid-meeting, read per utterance.
    pub language: String,
    /// Whether/how the meeting's ASR model takes a language hint.
    pub hint_support: super::language::HintSupport,
    pub stats: Stats,
    /// Every utterance's speech-end → text delay, for the end-of-meeting summary.
    asr_delays_ms: Vec<u32>,
}

#[derive(Debug, Default, Clone, Serialize, specta::Type)]
pub struct Stats {
    pub utterances: u32,
    /// Utterance end → words on screen (ASR compute + queueing), last and max.
    pub last_asr_delay_ms: u32,
    pub max_asr_delay_ms: u32,
    /// How far diarization trails the captured audio.
    pub diarizer_lag_ms: u32,
}

pub struct Session {
    pub shared: Arc<Mutex<Shared>>,
    asr: Option<JoinHandle<()>>,
    diar: Option<JoinHandle<()>>,
    settings: MeetingSettings,
    /// Set when audio comes from `MEETSCRIBE_SIMULATE_WAV` instead of the microphone.
    simulation: Option<(Arc<AtomicBool>, JoinHandle<()>)>,
}

/// What a finished meeting wrote.
#[derive(Debug, Clone, Default, Serialize, specta::Type)]
pub struct MeetingOutput {
    pub markdown_path: Option<String>,
    pub json_path: Option<String>,
    pub audio_path: Option<String>,
    pub error: Option<String>,
}

fn emit_turn_events(app: &AppHandle, shared: &mut Shared, events: Vec<TurnEvent>) {
    // display numbers follow first appearance in the transcript (see SpeakerMap)
    for ev in &events {
        if let TurnEvent::Appended { turn } | TurnEvent::Updated { turn } = ev {
            if let Some(speaker) = turn.speaker {
                shared.speakers.observe(speaker);
            }
        }
    }
    for ev in events {
        let (name, payload) = match ev {
            TurnEvent::Appended { turn } => (
                "meeting:turn_appended",
                super::TurnDto::from_turn(&turn, &shared.speakers),
            ),
            TurnEvent::Updated { turn } => (
                "meeting:turn_updated",
                super::TurnDto::from_turn(&turn, &shared.speakers),
            ),
            TurnEvent::Removed { id } => {
                let _ = app.emit("meeting:turn_removed", id);
                continue;
            }
        };
        let _ = app.emit(name, payload);
    }
}

fn emit_status(app: &AppHandle, shared: &Shared) {
    let _ = app.emit("meeting:status", super::MeetingStatus::running(shared));
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl Session {
    /// Starts the diarizer (waits for its model), loads the ASR model, then takes the microphone.
    pub fn start(
        app: &AppHandle,
        settings: MeetingSettings,
        diarizer: Box<dyn StreamingDiarizer>,
    ) -> Result<Self> {
        let tm = app.state::<Arc<TranscriptionManager>>().inner().clone();
        let rm = app.state::<Arc<AudioRecordingManager>>().inner().clone();

        if let Some(model) = &settings.asr_model {
            if tm.get_current_model().as_deref() != Some(model) {
                tm.load_model(model)
                    .context("failed to load the meeting ASR model")?;
            }
        } else if !tm.is_model_loaded() {
            tm.initiate_model_load();
        }
        let asr_model = tm
            .get_current_model()
            .or_else(|| settings.asr_model.clone())
            .unwrap_or_default();

        let hint_support = app
            .state::<Arc<crate::managers::model::ModelManager>>()
            .get_model_path(&asr_model)
            .map(|p| super::language::HintSupport::probe(&p))
            .unwrap_or_default();
        log::info!("Meeting ASR model {asr_model}: language hint support {hint_support:?}");

        let started = chrono::Local::now();
        let shared = Arc::new(Mutex::new(Shared {
            activity: SpeakerActivity::new(diarizer.num_speakers(), diarizer.frame_ms()),
            turns: TurnBuilder::new(TurnConfig::default()),
            speakers: SpeakerMap::new(),
            max_speakers: (settings.max_speakers as usize).min(diarizer.num_speakers()),
            started_local: started.format("%Y-%m-%d %H:%M").to_string(),
            started_epoch_ms: now_epoch_ms(),
            asr_model,
            diarizer: diarizer.describe(),
            diarizer_latency_ms: diarizer.latency_ms(),
            audio_ms: 0,
            language: settings.language.clone(),
            hint_support,
            stats: Stats::default(),
            asr_delays_ms: Vec::new(),
        }));

        let (diar_tx, diar_rx) = mpsc::channel::<Frame>();
        let (asr_tx, asr_rx) = mpsc::channel::<Frame>();

        let audio_path = settings.save_audio.then(|| {
            let s = shared.lock().unwrap();
            settings
                .resolved_output_dir(app)
                .join(format!("{}.wav", export::file_stem(&s.started_local)))
        });

        let diar = {
            let app = app.clone();
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("meeting-diarizer".into())
                .spawn(move || diarizer_worker(app, shared, diarizer, diar_rx, audio_path))?
        };
        let asr = {
            let app = app.clone();
            let shared = Arc::clone(&shared);
            let tm = Arc::clone(&tm);
            std::thread::Builder::new()
                .name("meeting-asr".into())
                .spawn(move || asr_worker(app, shared, tm, asr_rx))?
        };

        let clock = Arc::new(AtomicU64::new(0));
        let tap: crate::audio_toolkit::audio::AudioFrameCallback = {
            let (diar_tx, asr_tx) = (Mutex::new(diar_tx), Mutex::new(asr_tx));
            Arc::new(move |frame: &[f32]| {
                let start = clock.fetch_add(frame.len() as u64, Ordering::Relaxed);
                let _ = diar_tx.lock().unwrap().send(Frame {
                    start,
                    samples: frame.to_vec(),
                });
                let _ = asr_tx.lock().unwrap().send(Frame {
                    start,
                    samples: frame.to_vec(),
                });
            })
        };
        // On failure the tap (and its senders) is dropped, so both workers wind down.
        let simulation = match std::env::var_os("MEETSCRIBE_SIMULATE_WAV") {
            Some(path) => Some(simulate_from_file(app, PathBuf::from(path), tap)?),
            None => {
                rm.start_meeting_capture(MEETING_BINDING, tap)
                    .context("failed to open the microphone")?;
                None
            }
        };

        {
            let s = shared.lock().unwrap();
            emit_status(app, &s);
        }
        Ok(Self {
            shared,
            asr: Some(asr),
            diar: Some(diar),
            settings,
            simulation,
        })
    }

    /// Stops capture, drains both workers, finalizes the transcript and writes the outputs.
    pub fn stop(mut self, app: &AppHandle) -> MeetingOutput {
        if let Some((stop, feeder)) = self.simulation.take() {
            stop.store(true, Ordering::Release);
            let _ = feeder.join();
        } else {
            let rm = app.state::<Arc<AudioRecordingManager>>().inner().clone();
            // Dropping the sink drops the tap and closes both channels.
            let _ = rm.stop_recording(MEETING_BINDING, rm.cancel_generation());
        }
        if let Some(h) = self.asr.take() {
            let _ = h.join();
        }
        if let Some(h) = self.diar.take() {
            let _ = h.join();
        }

        let mut shared = self.shared.lock().unwrap();
        let events = {
            let Shared {
                turns, activity, ..
            } = &mut *shared;
            turns.finish(activity)
        };
        emit_turn_events(app, &mut shared, events);
        let mut delays = shared.asr_delays_ms.clone();
        delays.sort_unstable();
        let pct = |p: f64| {
            delays
                .get(((delays.len() as f64 - 1.0) * p).round() as usize)
                .copied()
                .unwrap_or(0)
        };
        log::info!(
            "Meeting stats: {:.1} min audio, {} utterances, speech end→text median {} ms / p95 {} ms / max {} ms",
            shared.audio_ms as f64 / 60_000.0,
            shared.stats.utterances,
            pct(0.5),
            pct(0.95),
            shared.stats.max_asr_delay_ms,
        );
        write_outputs(app, &self.settings, &shared)
    }
}

/// Dev/test hook: feeds a recording through the meeting pipeline in place of the microphone,
/// in 32 ms frames at `MEETSCRIBE_SIMULATE_SPEED`× real time (default 1). With
/// `MEETSCRIBE_SIMULATE_AUTOSTOP=1` the meeting stops itself when the file ends.
fn simulate_from_file(
    app: &AppHandle,
    path: PathBuf,
    tap: crate::audio_toolkit::audio::AudioFrameCallback,
) -> Result<(Arc<AtomicBool>, JoinHandle<()>)> {
    let audio = crate::audio_toolkit::audio::decode_audio_file(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let speed: f64 = std::env::var("MEETSCRIBE_SIMULATE_SPEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|s: &f64| *s > 0.0)
        .unwrap_or(1.0);
    let autostop = crate::utils::env_flag_enabled("MEETSCRIBE_SIMULATE_AUTOSTOP");
    log::info!(
        "Meeting audio simulated from {} ({:.1} s at {speed}x)",
        path.display(),
        audio.len() as f64 / SAMPLE_RATE as f64
    );
    let stop = Arc::new(AtomicBool::new(false));
    let handle = {
        let stop = Arc::clone(&stop);
        let app = app.clone();
        std::thread::Builder::new()
            .name("meeting-simulator".into())
            .spawn(move || {
                const FRAME: usize = 512;
                let frame_dur = Duration::from_secs_f64(FRAME as f64 / SAMPLE_RATE as f64 / speed);
                let t0 = Instant::now();
                for (i, frame) in audio.chunks(FRAME).enumerate() {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let due = t0 + frame_dur * i as u32;
                    if let Some(wait) = due.checked_duration_since(Instant::now()) {
                        std::thread::sleep(wait);
                    }
                    tap(frame);
                }
                drop(tap);
                if autostop && !stop.load(Ordering::Acquire) {
                    super::toggle(&app);
                }
            })?
    };
    Ok((stop, handle))
}

fn write_outputs(app: &AppHandle, settings: &MeetingSettings, shared: &Shared) -> MeetingOutput {
    let mut out = MeetingOutput::default();
    let dir = settings.resolved_output_dir(app);
    let stem = export::file_stem(&shared.started_local);
    let info = MeetingInfo {
        started_local: shared.started_local.clone(),
        duration_ms: shared.audio_ms,
        asr_model: shared.asr_model.clone(),
        diarizer_model: shared.diarizer.clone(),
        dropped_audio_ms: 0,
    };
    let result = (|| -> Result<()> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create {}", dir.display()))?;
        let md_path = unique_path(dir.join(format!("{stem}.md")));
        std::fs::write(
            &md_path,
            export::to_markdown(&info, shared.turns.turns(), &shared.speakers),
        )?;
        out.markdown_path = Some(md_path.to_string_lossy().into_owned());
        if settings.write_json {
            let json_path = md_path.with_extension("json");
            std::fs::write(
                &json_path,
                export::to_json(&info, shared.turns.turns(), &shared.speakers)?,
            )?;
            out.json_path = Some(json_path.to_string_lossy().into_owned());
        }
        if settings.save_audio {
            let wav = dir.join(format!("{stem}.wav"));
            if wav.is_file() {
                out.audio_path = Some(wav.to_string_lossy().into_owned());
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        log::error!("Failed to write meeting output: {e:#}");
        out.error = Some(format!("{e:#}"));
    }
    out
}

/// `path`, or `path` with ` (2)`, ` (3)`… if it exists (two meetings started in one minute).
fn unique_path(path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }
    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    (2..)
        .map(|i| path.with_file_name(format!("{stem} ({i}).{ext}")))
        .find(|p| !p.exists())
        .expect("unbounded")
}

fn apply_update(app: &AppHandle, shared: &Arc<Mutex<Shared>>, update: DiarUpdate, audio_end: u64) {
    let mut s = shared.lock().unwrap();
    s.audio_ms = samples_to_ms(audio_end);
    if !update.probs.is_empty() {
        let n = s.activity.num_speakers();
        let max = s.max_speakers;
        let mut probs = update.probs;
        if max < n {
            for frame in probs.chunks_mut(n) {
                frame[max..].fill(0.0);
            }
        }
        s.activity.write(update.start_frame, &probs);
    }
    s.activity.finalize_until(update.finalized_until);
    s.stats.diarizer_lag_ms = s.audio_ms.saturating_sub(s.activity.finalized_until_ms()) as u32;
    let events = {
        let Shared {
            turns, activity, ..
        } = &mut *s;
        turns.refresh(activity)
    };
    if !events.is_empty() {
        emit_turn_events(app, &mut s, events);
    }
}

fn diarizer_worker(
    app: AppHandle,
    shared: Arc<Mutex<Shared>>,
    mut diarizer: Box<dyn StreamingDiarizer>,
    rx: Receiver<Frame>,
    audio_path: Option<PathBuf>,
) {
    let mut wav = audio_path.and_then(|p| {
        let _ = std::fs::create_dir_all(p.parent()?);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        hound::WavWriter::create(&p, spec)
            .inspect_err(|e| log::error!("Cannot record meeting audio to {}: {e}", p.display()))
            .ok()
    });
    let mut audio_end = 0u64;
    let mut last_status = Instant::now();
    loop {
        let mut block = Vec::new();
        let mut disconnected = false;
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(f) => {
                audio_end = f.start + f.samples.len() as u64;
                block.extend_from_slice(&f.samples);
                while let Ok(f) = rx.try_recv() {
                    audio_end = f.start + f.samples.len() as u64;
                    block.extend_from_slice(&f.samples);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => disconnected = true,
        }
        if let Some(w) = wav.as_mut() {
            for &x in &block {
                let _ = w.write_sample((x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
            }
        }
        let update = if disconnected {
            diarizer.finish()
        } else {
            diarizer.push(&block)
        };
        match update {
            Ok(u) => apply_update(&app, &shared, u, audio_end),
            Err(e) => {
                log::error!("Diarizer failed: {e:#}");
                let _ = app.emit("meeting:error", format!("Diarizer failed: {e:#}"));
                // keep draining so capture never stalls; transcript continues without speakers
                for _ in rx.iter() {}
                break;
            }
        }
        if last_status.elapsed() > Duration::from_secs(1) {
            emit_status(&app, &shared.lock().unwrap());
            last_status = Instant::now();
        }
        if disconnected {
            break;
        }
    }
    if let Some(w) = wav {
        let _ = w.finalize();
    }
}

fn asr_worker(
    app: AppHandle,
    shared: Arc<Mutex<Shared>>,
    tm: Arc<TranscriptionManager>,
    rx: Receiver<Frame>,
) {
    let backend = crate::settings::get_settings(&app).vad_backend;
    let mut vad = match crate::managers::audio::create_detector(backend, &app) {
        Ok(v) => v,
        Err(e) => {
            log::error!("Meeting VAD unavailable: {e:#}");
            let _ = app.emit("meeting:error", format!("VAD unavailable: {e:#}"));
            for _ in rx.iter() {}
            return;
        }
    };
    let frame_samples = vad.frame_samples();
    let mut seg = Segmenter::new(SegmenterConfig::default(), frame_samples);
    let mut pending: Vec<f32> = Vec::new();
    let mut pending_start = 0u64;

    let handle = |u: Utterance| transcribe_utterance(&app, &shared, &tm, u);
    for f in rx.iter() {
        if pending.is_empty() {
            pending_start = f.start;
        }
        pending.extend_from_slice(&f.samples);
        let whole = pending.len() / frame_samples * frame_samples;
        if whole == 0 {
            continue;
        }
        let probs = vad.score(&pending[..whole]).unwrap_or_else(|e| {
            log::warn!("Meeting VAD failed on a block: {e:#}");
            vec![1.0; whole / frame_samples]
        });
        for (i, frame) in pending[..whole].chunks(frame_samples).enumerate() {
            let start = pending_start + (i * frame_samples) as u64;
            if let Some(u) = seg.push(start, frame, probs[i]) {
                handle(u);
            }
        }
        pending.drain(..whole);
        pending_start += whole as u64;
    }
    if let Some(u) = seg.flush() {
        handle(u);
    }
}

fn transcribe_utterance(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    tm: &TranscriptionManager,
    u: Utterance,
) {
    let end_sample = u.start_sample + u.pcm.len() as u64;
    let len_ms = samples_to_ms(u.pcm.len() as u64);
    let language = {
        let s = shared.lock().unwrap();
        s.hint_support.resolve(&s.language)
    };
    let started = Instant::now();
    let transcript = match tm.transcribe_timed(u.pcm, language) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("Meeting utterance transcription failed: {e:#}");
            return;
        }
    };
    let words = transcript_to_words(&transcript, u.start_sample, len_ms);
    let mut s = shared.lock().unwrap();
    s.stats.utterances += 1;
    // speech end → words in the transcript: audio captured since the utterance ended (the
    // segmenter's end-silence wait, queueing and ASR compute all show up here)
    let delay = s.audio_ms.saturating_sub(samples_to_ms(end_sample)) as u32;
    let compute = started.elapsed().as_millis() as u32;
    s.stats.last_asr_delay_ms = delay;
    s.stats.max_asr_delay_ms = s.stats.max_asr_delay_ms.max(delay);
    s.asr_delays_ms.push(delay);
    log::debug!(
        "meeting utterance {:.1}s → {} words in {} ms (forced split: {})",
        len_ms as f32 / 1000.0,
        words.len(),
        compute,
        u.forced_split
    );
    s.turns.add_words(words);
    let events = {
        let Shared {
            turns, activity, ..
        } = &mut *s;
        turns.refresh(activity)
    };
    emit_turn_events(app, &mut s, events);
}

/// Converts one utterance's transcript to meeting-timeline words: word rows when the engine has
/// them (Parakeet family), else segment rows (Whisper), else one span for the whole text.
pub fn transcript_to_words(t: &Transcript, start_sample: u64, utterance_ms: u64) -> Vec<TimedWord> {
    let clamp = |ms: i64| ms.clamp(0, utterance_ms as i64);
    let to_word = |t0: i64, t1: i64, text: &str| {
        TimedWord::new(
            utterance_to_meeting_ms(start_sample, clamp(t0)),
            utterance_to_meeting_ms(start_sample, clamp(t1)),
            text.trim(),
        )
    };
    let words: Vec<TimedWord> = if !t.words.is_empty() {
        t.words
            .iter()
            .map(|w| to_word(w.t0_ms, w.t1_ms, &w.text))
            .collect()
    } else if !t.segments.is_empty() {
        t.segments
            .iter()
            .map(|s| to_word(s.t0_ms, s.t1_ms, &s.text))
            .collect()
    } else if !t.text.trim().is_empty() {
        vec![to_word(0, utterance_ms as i64, &t.text)]
    } else {
        Vec::new()
    };
    words.into_iter().filter(|w| !w.text.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcribe_cpp::{Segment, Word};

    fn word(t0: i64, t1: i64, text: &str) -> Word {
        Word {
            t0_ms: t0,
            t1_ms: t1,
            text: text.into(),
            ..Default::default()
        }
    }

    #[test]
    fn words_are_shifted_onto_the_meeting_clock() {
        let t = Transcript {
            words: vec![word(0, 300, " goeie"), word(300, 700, "morgen")],
            ..Default::default()
        };
        // utterance starts at 10 s
        let w = transcript_to_words(&t, 160_000, 1_000);
        assert_eq!(
            w,
            vec![
                TimedWord::new(10_000, 10_300, "goeie"),
                TimedWord::new(10_300, 10_700, "morgen")
            ]
        );
    }

    #[test]
    fn segment_fallback_and_clamping() {
        let t = Transcript {
            segments: vec![Segment {
                t0_ms: -20,
                t1_ms: 5_000,
                text: " hallo allemaal ".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let w = transcript_to_words(&t, 16_000, 2_000);
        assert_eq!(w, vec![TimedWord::new(1_000, 3_000, "hallo allemaal")]);
    }

    #[test]
    fn text_only_engine_spans_the_utterance() {
        let t = Transcript {
            text: "oké".into(),
            ..Default::default()
        };
        assert_eq!(
            transcript_to_words(&t, 0, 800),
            vec![TimedWord::new(0, 800, "oké")]
        );
        assert!(transcript_to_words(&Transcript::default(), 0, 800).is_empty());
    }
}
