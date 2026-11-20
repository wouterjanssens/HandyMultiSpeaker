//! One running meeting: capture tap → diarizer worker + ASR worker → shared transcript state.
//!
//! Timeline: the capture tap stamps every 16 kHz frame with its absolute sample index (one
//! counter, pre-VAD). The diarizer receives every sample from index 0, so its frame `i` covers
//! `[i*10ms, (i+1)*10ms)` of the same timeline. The ASR worker runs its own VAD/segmenter on the
//! same stamped frames and shifts each transcript's word times by the utterance's start sample.
//! Known gap: frames dropped by a capture-ring overrun are not visible per chunk, so such a gap
//! shifts everything after it (both workers equally); it is logged at stop.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

use super::postprocess::{self, Finish};
use super::settings::{MeetingSettings, PostProcessMode};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
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
    /// Changeable mid-meeting; read by the post-processing worker.
    pub post_process_mode: PostProcessMode,
    pub post_process_prompt_id: Option<String>,
    /// LLM-cleaned text of locked turns, by turn id.
    pub cleaned: HashMap<u32, String>,
    /// meetscribe: utterance spans (ms), for cutting the recording at pauses (OpenAI final pass).
    pub utterance_spans: Vec<(u64, u64)>,
    /// meetscribe: file name (no extension) all of this meeting's files use, reserved at start.
    pub stem: String,
    /// meetscribe: the live `.vtt` transcript; `None` once the meeting has ended (or when the
    /// file could not be created).
    pub live: Option<super::live::LiveVtt>,
}

/// meetscribe: what a meeting that ran on OpenAI still needs once it has stopped (the final
/// speaker pass and the notes). Carried to the post-processing worker, so Stop never waits for
/// it. Only meetings on the OpenAI model have one.
pub struct CloudFinish {
    client: crate::openai::Client,
    settings: crate::openai::OpenAiSettings,
    keywords: Vec<String>,
    /// The recording for the final speaker pass; `None` when that pass is off.
    wav: Option<PathBuf>,
    /// The recording is the user's "save audio" file; otherwise it is deleted afterwards.
    keep_wav: bool,
}

/// Prefix of the temporary recording the final speaker pass uses without "save audio".
const TEMP_WAV_PREFIX: &str = "meetscribe-";

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
    cloud_finish: Option<CloudFinish>,
    /// Set when audio comes from `MEETSCRIBE_SIMULATE_WAV` instead of the microphone.
    simulation: Option<(Arc<AtomicBool>, JoinHandle<()>)>,
    /// Hands the finished meeting to the post-processing worker.
    post_process: mpsc::Sender<Finish>,
}

/// What a finished meeting wrote.
#[derive(Debug, Clone, Default, Serialize, specta::Type)]
pub struct MeetingOutput {
    pub markdown_path: Option<String>,
    pub json_path: Option<String>,
    pub audio_path: Option<String>,
    /// AI meeting notes (OpenAI), when enabled.
    pub notes_path: Option<String>,
    /// meetscribe: the WebVTT transcript (live during the meeting, final afterwards).
    pub vtt_path: Option<String>,
    pub error: Option<String>,
}

impl Shared {
    // meetscribe: used by the OpenAI pipeline (`cloud.rs`)
    pub(crate) fn push_asr_delay(&mut self, delay_ms: u32) {
        self.asr_delays_ms.push(delay_ms);
    }

    /// Re-attributes open words against the current activity and emits the UI events.
    pub(crate) fn refresh_and_emit(&mut self, app: &AppHandle) {
        let events = {
            let Shared {
                turns, activity, ..
            } = &mut *self;
            turns.refresh(activity)
        };
        if !events.is_empty() {
            emit_turn_events(app, self, events);
        }
    }
}

pub(crate) fn emit_turn_events(app: &AppHandle, shared: &mut Shared, events: Vec<TurnEvent>) {
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
                super::TurnDto::from_turn(&turn, shared),
            ),
            TurnEvent::Updated { turn } => (
                "meeting:turn_updated",
                super::TurnDto::from_turn(&turn, shared),
            ),
            TurnEvent::Removed { id } => {
                let _ = app.emit("meeting:turn_removed", id);
                continue;
            }
        };
        let _ = app.emit(name, payload);
    }
    // meetscribe: finished turns go to the live transcript file
    let Shared {
        live,
        turns,
        speakers,
        ..
    } = shared;
    if let Some(live) = live.as_mut() {
        live.sync(turns, speakers);
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
        cloud: Option<super::cloud::CloudConfig>,
    ) -> Result<Self> {
        let tm = app.state::<Arc<TranscriptionManager>>().inner().clone();
        let rm = app.state::<Arc<AudioRecordingManager>>().inner().clone();

        if let Some(c) = &cloud {
            // meetscribe: OpenAI does the transcription; no local model to load
            log::info!(
                "Meeting on OpenAI: {} + {}",
                c.settings.transcribe_model,
                c.settings.diarize_model
            );
        } else if let Some(model) = &settings.asr_model {
            if tm.get_current_model().as_deref() != Some(model) {
                tm.load_model(model)
                    .context("failed to load the meeting ASR model")?;
            }
        } else if !tm.is_model_loaded() {
            tm.initiate_model_load();
        }
        let (asr_model, hint_support) = match &cloud {
            Some(c) => (
                format!("{} (OpenAI)", c.settings.transcribe_model),
                super::language::HintSupport {
                    honored: Some(true),
                    codes: super::settings::MEETING_LANGUAGES
                        .iter()
                        .map(|l| l.to_string())
                        .collect(),
                },
            ),
            None => {
                let asr_model = tm
                    .get_current_model()
                    .or_else(|| settings.asr_model.clone())
                    .unwrap_or_default();
                let hint_support = app
                    .state::<Arc<crate::managers::model::ModelManager>>()
                    .get_model_path(&asr_model)
                    .map(|p| super::language::HintSupport::probe(&p))
                    .unwrap_or_default();
                (asr_model, hint_support)
            }
        };
        log::info!("Meeting ASR model {asr_model}: language hint support {hint_support:?}");

        let started = chrono::Local::now();
        // meetscribe: one file name for everything this meeting writes, reserved now so the live
        // transcript and the final files match
        let output_dir = settings.resolved_output_dir(app);
        let started_label = started.format("%Y-%m-%d %H:%M").to_string();
        let stem = reserve_stem(&output_dir, &export::file_stem(&started_label));
        let live = match super::live::LiveVtt::create(&output_dir, &stem, &started_label) {
            Ok(live) => {
                log::info!("Live transcript: {}", live.path().display());
                Some(live)
            }
            Err(e) => {
                log::warn!("No live transcript file: {e:#}");
                None
            }
        };
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
            post_process_mode: settings.post_process_mode,
            post_process_prompt_id: settings.post_process_prompt_id.clone(),
            cleaned: HashMap::new(),
            utterance_spans: Vec::new(),
            stem,
            live,
        }));

        let (diar_tx, diar_rx) = mpsc::channel::<Frame>();
        let (asr_tx, asr_rx) = mpsc::channel::<Frame>();

        let audio_path = settings.save_audio.then(|| {
            let s = shared.lock().unwrap();
            settings
                .resolved_output_dir(app)
                .join(format!("{}.wav", s.stem))
        });
        // meetscribe: the OpenAI final speaker pass needs the recording; without "save audio"
        // it goes to a temporary file
        let cloud_finish = cloud.as_ref().map(|c| CloudFinish {
            client: c.client.clone(),
            settings: c.settings.clone(),
            keywords: c.keywords.clone(),
            wav: c.settings.final_pass.then(|| {
                audio_path.clone().unwrap_or_else(|| {
                    let s = shared.lock().unwrap();
                    std::env::temp_dir().join(format!(
                        "{TEMP_WAV_PREFIX}{}-{}.wav",
                        export::file_stem(&s.started_local),
                        std::process::id()
                    ))
                })
            }),
            keep_wav: audio_path.is_some(),
        });
        let audio_path = audio_path.or_else(|| cloud_finish.as_ref().and_then(|c| c.wav.clone()));

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
                .spawn(move || asr_worker(app, shared, tm, asr_rx, cloud))?
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

        let post_process = postprocess::spawn(app, Arc::clone(&shared))?;
        {
            let s = shared.lock().unwrap();
            emit_status(app, &s);
        }
        Ok(Self {
            post_process,
            shared,
            asr: Some(asr),
            diar: Some(diar),
            settings,
            cloud_finish,
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
        // meetscribe: the live file is complete; the final pass must not append to it
        let live_vtt = shared
            .live
            .take()
            .map(|live| live.end(&shared.turns, &shared.speakers));
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
        let info = meeting_info(&shared);
        let output = write_outputs(app, &self.settings, &shared, &info);
        let history_id = add_to_history(app, &shared, &output);
        // The worker takes it from here (final speaker pass, final `.vtt`, notes, clean-up,
        // rename); it is gone only if it panicked.
        let _ = self.post_process.send(Finish {
            info,
            output: output.clone(),
            history_id,
            cloud: self.cloud_finish.take(),
            live_vtt,
        });
        output
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

fn meeting_info(shared: &Shared) -> MeetingInfo {
    MeetingInfo {
        started_local: shared.started_local.clone(),
        duration_ms: shared.audio_ms,
        asr_model: shared.asr_model.clone(),
        diarizer_model: shared.diarizer.clone(),
        dropped_audio_ms: 0,
    }
}

fn write_outputs(
    app: &AppHandle,
    settings: &MeetingSettings,
    shared: &Shared,
    info: &MeetingInfo,
) -> MeetingOutput {
    let mut out = MeetingOutput::default();
    let dir = settings.resolved_output_dir(app);
    // meetscribe: the name reserved at start, shared with the live `.vtt`
    let stem = shared.stem.clone();
    let result = (|| -> Result<()> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create {}", dir.display()))?;
        let md_path = dir.join(format!("{stem}.md"));
        std::fs::write(
            &md_path,
            export::to_markdown(info, shared.turns.turns(), &shared.speakers),
        )?;
        out.markdown_path = Some(md_path.to_string_lossy().into_owned());
        if settings.write_json {
            let json_path = md_path.with_extension("json");
            std::fs::write(
                &json_path,
                export::to_json(info, shared.turns.turns(), &shared.speakers)?,
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

/// meetscribe: what has to happen to a stopped meeting before post-processing: the OpenAI final
/// speaker pass (then the saved transcript and History entry are rewritten), the final `.vtt`,
/// and the OpenAI notes. Runs on the post-processing worker, never under the session's stop.
/// Returns true when the turns were replaced (their ids changed).
pub(super) fn finish_after_stop(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    finish: &mut Finish,
) -> bool {
    let mut replaced = false;
    if let Some(cloud) = &finish.cloud {
        if let Some(wav) = &cloud.wav {
            replaced = run_final_pass(app, shared, cloud, wav);
            if replaced {
                rewrite_outputs(app, shared, finish);
            }
            if !cloud.keep_wav {
                let _ = std::fs::remove_file(wav);
            }
        }
    }
    if let Some(path) = finish.live_vtt.take() {
        // (no file name in it: post-processing may still rename the meeting's files)
        let document = {
            let s = shared.lock().unwrap();
            meeting_core::vtt::final_document(&s.started_local, s.turns.turns(), &s.speakers, None)
        };
        super::live::finalize(&path, &document);
        finish.output.vtt_path = Some(path.to_string_lossy().into_owned());
    }
    if let Some(cloud) = finish.cloud.as_ref().filter(|c| c.settings.notes) {
        finish.output.notes_path = write_notes(app, shared, cloud, &finish.output);
    }
    // the meeting window shows the files; tell it about the ones added since Stop, unless a new
    // meeting has started in the window since
    if super::phase(app) == super::MeetingPhase::Idle {
        let _ = app.emit("meeting:stopped", finish.output.clone());
    }
    replaced
}

/// meetscribe: replaces the live speaker labels with the OpenAI full-recording pass. The lock is
/// only held to copy the input and to swap in the result. On failure the live transcript is kept.
fn run_final_pass(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    cloud: &CloudFinish,
    wav: &Path,
) -> bool {
    let (live, spans, language) = {
        let s = shared.lock().unwrap();
        let live: Vec<meeting_core::AttributedWord> = s
            .turns
            .turns()
            .flat_map(|t| t.words.iter().cloned())
            .collect();
        (live, s.utterance_spans.clone(), s.language.clone())
    };
    if spans.is_empty() {
        return false;
    }
    let languages = if language == "auto" {
        cloud.settings.languages.clone()
    } else {
        vec![language]
    };
    let started = Instant::now();
    let refined = match super::refine::run(
        &cloud.client,
        &cloud.settings,
        wav,
        &spans,
        &live,
        &languages,
        &cloud.keywords,
    ) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Final speaker pass failed, keeping live speakers: {e:#}");
            let _ = app.emit(
                "meeting:error",
                format!("Final speaker pass failed (live speakers kept): {e:#}"),
            );
            return false;
        }
    };
    let mut s = shared.lock().unwrap();
    let old_ids: Vec<u32> = s.turns.turns().map(|t| t.id).collect();
    let mut turns = TurnBuilder::new(TurnConfig::default());
    turns.add_words(refined.words);
    turns.finish(&refined.activity);
    s.turns = turns;
    s.activity = refined.activity;
    // cleaned text belongs to the old turns
    s.cleaned.clear();
    for id in old_ids.into_iter().rev() {
        let _ = app.emit("meeting:turn_removed", id);
    }
    let appended: Vec<TurnEvent> = s
        .turns
        .turns()
        .map(|t| TurnEvent::Appended { turn: t.clone() })
        .collect();
    emit_turn_events(app, &mut s, appended);
    log::info!(
        "Final speaker pass in {:.1}s: {} speakers, {} spans split",
        started.elapsed().as_secs_f64(),
        refined.speakers,
        refined.splits
    );
    true
}

/// meetscribe: writes the transcript and History entry again after the final speaker pass, at the
/// paths saved at Stop.
fn rewrite_outputs(app: &AppHandle, shared: &Arc<Mutex<Shared>>, finish: &Finish) {
    let s = shared.lock().unwrap();
    let result = (|| -> Result<()> {
        if let Some(md) = &finish.output.markdown_path {
            std::fs::write(
                md,
                export::to_markdown(&finish.info, s.turns.turns(), &s.speakers),
            )?;
        }
        if let Some(json) = &finish.output.json_path {
            std::fs::write(
                json,
                export::to_json(&finish.info, s.turns.turns(), &s.speakers)?,
            )?;
        }
        Ok(())
    })();
    if let Err(e) = result {
        log::error!("Failed to rewrite the meeting after the speaker pass: {e:#}");
    }
    if let Some(id) = finish.history_id {
        let text = export::transcript_text(s.turns.turns(), &s.speakers);
        if let Err(e) = app
            .state::<Arc<crate::managers::history::HistoryManager>>()
            .update_transcription(id, text.trim_end().to_string(), None, None)
        {
            log::error!("Failed to update the meeting in history: {e:#}");
        }
    }
}

/// meetscribe: AI meeting notes next to the transcript (`…_notes.md`). Failures are logged and
/// reported, never fatal.
fn write_notes(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    cloud: &CloudFinish,
    output: &MeetingOutput,
) -> Option<String> {
    let (started_local, utterances) = {
        let s = shared.lock().unwrap();
        (s.started_local.clone(), s.stats.utterances)
    };
    let md_path = PathBuf::from(output.markdown_path.as_ref()?);
    if utterances == 0 {
        return None;
    }
    let transcript = std::fs::read_to_string(&md_path).ok()?;
    let started = Instant::now();
    let model = &cloud.settings.notes_model;
    match crate::openai::notes::generate(&cloud.client, model, &transcript) {
        Ok(notes) => {
            let stem = md_path.file_stem()?.to_string_lossy().into_owned();
            let path = md_path.with_file_name(format!("{stem}_notes.md"));
            let doc = format!(
                "# Notes – {started_local}\n\n_Generated by {model} from [{}]({})._\n\n{notes}\n",
                md_path.file_name()?.to_string_lossy(),
                md_path.file_name()?.to_string_lossy().replace(' ', "%20"),
            );
            match std::fs::write(&path, doc) {
                Ok(()) => {
                    log::info!(
                        "Meeting notes written in {:.1}s: {}",
                        started.elapsed().as_secs_f64(),
                        path.display()
                    );
                    Some(path.to_string_lossy().into_owned())
                }
                Err(e) => {
                    log::error!("Cannot write meeting notes: {e}");
                    None
                }
            }
        }
        Err(e) => {
            log::error!("Meeting notes failed: {e:#}");
            let _ = app.emit("meeting:error", format!("Meeting notes failed: {e:#}"));
            None
        }
    }
}

/// meetscribe: removes final-pass recordings left in the temp folder by a crash. Called at
/// startup, when no meeting can be running.
pub fn remove_stale_recordings() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(TEMP_WAV_PREFIX) && name.ends_with(".wav") {
            match std::fs::remove_file(entry.path()) {
                Ok(()) => log::info!("Removed a stale meeting recording: {name}"),
                Err(e) => log::warn!("Cannot remove {name}: {e}"),
            }
        }
    }
}

/// Lists the saved meeting in History and returns the entry's id. A meeting in which nothing was
/// said isn't worth a row.
fn add_to_history(app: &AppHandle, shared: &Shared, output: &MeetingOutput) -> Option<i64> {
    let transcript_path = output.markdown_path.as_ref()?;
    let transcript = export::transcript_text(shared.turns.turns(), &shared.speakers);
    let transcript = transcript.trim_end();
    if transcript.is_empty() {
        return None;
    }
    app.state::<Arc<HistoryManager>>()
        .save_meeting_entry(
            (shared.started_epoch_ms / 1000) as i64,
            transcript.to_string(),
            Path::new(transcript_path),
            output.audio_path.as_deref().map(Path::new),
        )
        .inspect_err(|e| log::error!("Failed to add the meeting to history: {e:#}"))
        .ok()
        .map(|entry| entry.id)
}

/// `path`, or `path` with ` (2)`, ` (3)`… if it exists (two meetings started in one minute).
pub(super) fn unique_path(path: PathBuf) -> PathBuf {
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
        .unwrap_or(path)
}

/// meetscribe: a file stem none of whose files (`.md`, `.vtt`) exist yet: `stem`, `stem (2)`, …
fn reserve_stem(dir: &Path, stem: &str) -> String {
    let free = |s: &str| {
        ["md", "vtt"]
            .iter()
            .all(|ext| !dir.join(format!("{s}.{ext}")).exists())
    };
    if free(stem) {
        return stem.to_string();
    }
    (2..)
        .map(|i| format!("{stem} ({i})"))
        .find(|s| free(s))
        .unwrap_or_else(|| stem.to_string())
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
    cloud: Option<super::cloud::CloudConfig>,
) {
    // meetscribe: with OpenAI, utterances go to the cloud pipeline instead of the local model
    let cloud = cloud.map(|cfg| {
        let (tx, utt_rx) = mpsc::channel::<Utterance>();
        let (app, shared) = (app.clone(), Arc::clone(&shared));
        let worker = std::thread::Builder::new()
            .name("meeting-openai-text".into())
            .spawn(move || super::cloud::run(app, shared, cfg, utt_rx))
            .expect("spawn OpenAI meeting pipeline");
        (tx, worker)
    });
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

    let handle = |u: Utterance| match &cloud {
        Some((tx, _)) => {
            let _ = tx.send(u);
        }
        None => transcribe_utterance(&app, &shared, &tm, u),
    };
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
    if let Some((tx, worker)) = cloud {
        drop(tx);
        let _ = worker.join();
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
