//! MeetScribe meeting mode: live, speaker-attributed transcription of a whole meeting.
//!
//! Entry points: [`toggle`] (shortcut, tray, `--toggle-meeting`), the Tauri commands below, and
//! [`is_active`], which dictation paths consult so they never touch the microphone a meeting
//! holds. Pure logic lives in the `meeting-core` crate; this module wires it to audio capture,
//! the ASR engine, a diarizer backend, the meeting window and the output files.

pub mod model;
pub mod session;
pub mod settings;
pub mod sidecar;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use meeting_core::{SpeakerMap, StreamingDiarizer, Turn};
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};

use session::{MeetingOutput, Session, Shared, Stats};
use settings::{DiarizerBackend, MeetingSettings};

pub use session::MEETING_BINDING;

pub const WINDOW_LABEL: &str = "meeting";

enum State {
    Idle,
    /// Loading models / opening the microphone.
    Starting,
    Running(Session),
    /// Draining workers and writing files.
    Stopping,
}

pub struct MeetingManager {
    state: Mutex<State>,
}

impl MeetingManager {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::Idle),
        }
    }
}

impl Default for MeetingManager {
    fn default() -> Self {
        Self::new()
    }
}

fn manager(app: &AppHandle) -> Option<Arc<MeetingManager>> {
    app.try_state::<Arc<MeetingManager>>()
        .map(|s| s.inner().clone())
}

/// Current phase (Idle when meeting mode isn't initialised, e.g. headless runs).
pub fn phase(app: &AppHandle) -> MeetingPhase {
    manager(app)
        .map(|m| match &*m.state.lock().unwrap() {
            State::Idle => MeetingPhase::Idle,
            State::Starting => MeetingPhase::Starting,
            State::Running(_) => MeetingPhase::Running,
            State::Stopping => MeetingPhase::Stopping,
        })
        .unwrap_or(MeetingPhase::Idle)
}

/// True while a meeting is starting, running or finishing; dictation must stay off the mic.
pub fn is_active(app: &AppHandle) -> bool {
    phase(app) != MeetingPhase::Idle
}

/// A turn as the meeting window renders it.
#[derive(Debug, Clone, Serialize, Type)]
pub struct TurnDto {
    pub id: u32,
    pub speaker: Option<u8>,
    pub label: String,
    pub start_ms: f64,
    pub end_ms: f64,
    pub text: String,
    pub locked: bool,
}

impl TurnDto {
    pub fn from_turn(t: &Turn, speakers: &SpeakerMap) -> Self {
        Self {
            id: t.id,
            speaker: t.speaker,
            label: speakers.label(t.speaker),
            start_ms: t.start_ms as f64,
            end_ms: t.end_ms as f64,
            text: t.text.clone(),
            locked: t.locked,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MeetingPhase {
    Idle,
    Starting,
    Running,
    Stopping,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct SpeakerDto {
    pub id: u8,
    pub label: String,
    pub custom_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct MeetingStatus {
    pub phase: MeetingPhase,
    pub started_epoch_ms: Option<f64>,
    pub audio_ms: f64,
    pub speakers: Vec<SpeakerDto>,
    pub asr_model: Option<String>,
    pub diarizer: Option<String>,
    pub diarizer_latency_ms: f64,
    pub stats: Option<Stats>,
}

impl MeetingStatus {
    fn idle(phase: MeetingPhase) -> Self {
        Self {
            phase,
            started_epoch_ms: None,
            audio_ms: 0.0,
            speakers: Vec::new(),
            asr_model: None,
            diarizer: None,
            diarizer_latency_ms: 0.0,
            stats: None,
        }
    }

    pub(crate) fn running(s: &Shared) -> Self {
        let ids: Vec<u8> = s.speakers.seen().to_vec();
        Self {
            phase: MeetingPhase::Running,
            started_epoch_ms: Some(s.started_epoch_ms as f64),
            audio_ms: s.audio_ms as f64,
            speakers: ids
                .into_iter()
                .map(|id| SpeakerDto {
                    id,
                    label: s.speakers.label(Some(id)),
                    custom_name: s.speakers.custom_name(id).map(str::to_string),
                })
                .collect(),
            asr_model: Some(s.asr_model.clone()),
            diarizer: Some(s.diarizer.clone()),
            diarizer_latency_ms: s.diarizer_latency_ms as f64,
            stats: Some(s.stats.clone()),
        }
    }
}

fn set_phase(app: &AppHandle, m: &MeetingManager, state: State) {
    let phase = match &state {
        State::Idle => MeetingPhase::Idle,
        State::Starting => MeetingPhase::Starting,
        State::Running(_) => MeetingPhase::Running,
        State::Stopping => MeetingPhase::Stopping,
    };
    *m.state.lock().unwrap() = state;
    if phase != MeetingPhase::Running {
        let _ = app.emit("meeting:status", MeetingStatus::idle(phase));
    }
    crate::tray::update_tray_menu(app);
}

fn create_diarizer(
    app: &AppHandle,
    settings: &MeetingSettings,
) -> anyhow::Result<Box<dyn StreamingDiarizer>> {
    let profile = settings.latency_profile.sidecar_name();
    let native = match settings.diarizer_backend {
        DiarizerBackend::Auto => diarize::AVAILABLE,
        DiarizerBackend::Native => true,
        DiarizerBackend::PythonSidecar => false,
    };
    if native {
        let path = model::ensure_diarizer_model(app)?;
        let cfg = diarize::NativeConfig::for_profile(profile);
        Ok(Box::new(diarize::NativeDiarizer::new(&path, &cfg)?))
    } else {
        Ok(Box::new(sidecar::PythonSidecarDiarizer::spawn(
            profile,
            "auto",
            Duration::from_secs(300),
        )?))
    }
}

/// Starts a meeting if none is running, otherwise stops it. Never blocks the caller.
pub fn toggle(app: &AppHandle) {
    let Some(m) = manager(app) else { return };
    let current = phase(app);
    match current {
        MeetingPhase::Idle => start(app, m),
        MeetingPhase::Running => stop(app, m),
        _ => log::info!("Meeting toggle ignored while {current:?}"),
    }
}

fn start(app: &AppHandle, m: Arc<MeetingManager>) {
    if crate::managers::audio::AudioRecordingManager::is_recording(
        &app.state::<Arc<crate::managers::audio::AudioRecordingManager>>(),
    ) {
        let _ = app.emit(
            "meeting:error",
            "Finish the current dictation before starting a meeting.",
        );
        return;
    }
    set_phase(app, &m, State::Starting);
    show_window(app);
    let app = app.clone();
    std::thread::spawn(move || {
        let settings = settings::load(&app);
        let result =
            create_diarizer(&app, &settings).and_then(|d| Session::start(&app, settings, d));
        match result {
            Ok(session) => {
                log::info!("Meeting started");
                set_phase(&app, &m, State::Running(session));
            }
            Err(e) => {
                log::error!("Could not start meeting: {e:#}");
                set_phase(&app, &m, State::Idle);
                let _ = app.emit(
                    "meeting:error",
                    format!("Could not start the meeting: {e:#}"),
                );
            }
        }
    });
}

fn stop(app: &AppHandle, m: Arc<MeetingManager>) {
    let session = {
        let mut guard = m.state.lock().unwrap();
        match std::mem::replace(&mut *guard, State::Stopping) {
            State::Running(s) => s,
            other => {
                *guard = other;
                return;
            }
        }
    };
    set_phase(app, &m, State::Stopping);
    let app = app.clone();
    std::thread::spawn(move || {
        let output: MeetingOutput = session.stop(&app);
        log::info!("Meeting stopped: {output:?}");
        set_phase(&app, &m, State::Idle);
        let _ = app.emit("meeting:stopped", output);
    });
}

/// Opens (or focuses) the meeting window.
pub fn show_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window(WINDOW_LABEL) {
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    let builder = WebviewWindowBuilder::new(
        app,
        WINDOW_LABEL,
        WebviewUrl::App("src/meeting/index.html".into()),
    )
    .title("MeetScribe – Meeting")
    .inner_size(760.0, 820.0)
    .min_inner_size(420.0, 400.0)
    .resizable(true);
    let builder = match crate::portable::data_dir() {
        Some(dir) => builder.data_directory(dir.join("webview")),
        None => builder,
    };
    if let Err(e) = builder.build() {
        log::error!("Failed to open meeting window: {e}");
    }
}

/* ---------- commands ---------- */

#[tauri::command]
#[specta::specta]
pub fn meeting_toggle(app: AppHandle) {
    toggle(&app);
}

#[tauri::command]
#[specta::specta]
pub fn meeting_open_window(app: AppHandle) {
    show_window(&app);
}

#[tauri::command]
#[specta::specta]
pub fn meeting_status(app: AppHandle) -> MeetingStatus {
    let Some(m) = manager(&app) else {
        return MeetingStatus::idle(MeetingPhase::Idle);
    };
    let guard = m.state.lock().unwrap();
    match &*guard {
        State::Idle => MeetingStatus::idle(MeetingPhase::Idle),
        State::Starting => MeetingStatus::idle(MeetingPhase::Starting),
        State::Stopping => MeetingStatus::idle(MeetingPhase::Stopping),
        State::Running(s) => MeetingStatus::running(&s.shared.lock().unwrap()),
    }
}

/// Current transcript (for a window opened mid-meeting).
#[tauri::command]
#[specta::specta]
pub fn meeting_turns(app: AppHandle) -> Vec<TurnDto> {
    let Some(m) = manager(&app) else {
        return Vec::new();
    };
    let guard = m.state.lock().unwrap();
    match &*guard {
        State::Running(s) => {
            let s = s.shared.lock().unwrap();
            s.turns
                .turns()
                .map(|t| TurnDto::from_turn(t, &s.speakers))
                .collect()
        }
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct SpeakerRenamed {
    pub speaker: u8,
    pub label: String,
}

/// Names a speaker; applies to the whole transcript, including the saved file.
#[tauri::command]
#[specta::specta]
pub fn meeting_rename_speaker(app: AppHandle, speaker: u8, name: String) -> Result<String, String> {
    let m = manager(&app).ok_or("meeting mode unavailable")?;
    let guard = m.state.lock().unwrap();
    let State::Running(session) = &*guard else {
        return Err("No meeting is running".into());
    };
    let mut s = session.shared.lock().unwrap();
    s.speakers.rename(speaker, &name);
    let label = s.speakers.label(Some(speaker));
    let _ = app.emit(
        "meeting:speaker_renamed",
        SpeakerRenamed {
            speaker,
            label: label.clone(),
        },
    );
    Ok(label)
}

/// Opens the output folder in Finder / Explorer.
#[tauri::command]
#[specta::specta]
pub fn meeting_reveal_output(app: AppHandle, path: Option<String>) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let target = match path {
        Some(p) => std::path::PathBuf::from(p),
        None => settings::load(&app).resolved_output_dir(&app),
    };
    if target.is_file() {
        app.opener()
            .reveal_item_in_dir(&target)
            .map_err(|e| e.to_string())
    } else {
        std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
        app.opener()
            .open_path(target.to_string_lossy(), None::<&str>)
            .map_err(|e| e.to_string())
    }
}
