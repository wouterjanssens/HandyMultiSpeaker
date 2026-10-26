//! Meeting-mode settings, kept in their own store file so upstream's `AppSettings` (and its
//! migrations) stay untouched.

use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Manager};
use tauri_plugin_store::StoreExt;

const STORE_PATH: &str = "meeting_settings.json";
const STORE_KEY: &str = "meeting";

/// Diarizer streaming latency profile (input buffer before a chunk is labelled).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LatencyProfile {
    /// 0.32 s buffer, 0.24 s hop.
    UltraLow,
    /// 0.64 s buffer, 0.48 s hop.
    VeryLow,
    /// 1.04 s buffer, 0.72 s hop (recommended; best accuracy/latency trade-off in the spike).
    Low,
    /// 5.12 s buffer, 4.48 s hop.
    High,
}

impl LatencyProfile {
    /// Name understood by `research/diarize_server.py`.
    pub fn sidecar_name(self) -> &'static str {
        match self {
            Self::UltraLow => "ultra_low_latency",
            Self::VeryLow => "very_low_latency",
            Self::Low => "low_latency",
            Self::High => "p5s",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DiarizerBackend {
    /// Temporary: Hugging Face reference implementation in a Python process (dev builds only).
    PythonSidecar,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct MeetingSettings {
    /// Speakers the transcript may use (1–8). Diarizer channels beyond this are ignored.
    pub max_speakers: u8,
    pub latency_profile: LatencyProfile,
    /// Folder for transcripts; `None` = `~/Documents/MeetScribe`.
    pub output_dir: Option<String>,
    /// Also keep the meeting audio as a WAV next to the transcript.
    pub save_audio: bool,
    /// ASR model for meetings; `None` = the model selected for dictation.
    pub asr_model: Option<String>,
    pub write_json: bool,
    pub diarizer_backend: DiarizerBackend,
}

impl Default for MeetingSettings {
    fn default() -> Self {
        Self {
            max_speakers: 6,
            latency_profile: LatencyProfile::Low,
            output_dir: None,
            save_audio: false,
            asr_model: None,
            write_json: true,
            diarizer_backend: DiarizerBackend::PythonSidecar,
        }
    }
}

impl MeetingSettings {
    fn sanitized(mut self) -> Self {
        self.max_speakers = self.max_speakers.clamp(1, 8);
        self.output_dir = self.output_dir.filter(|d| !d.trim().is_empty());
        self.asr_model = self.asr_model.filter(|m| !m.trim().is_empty());
        self
    }

    pub fn resolved_output_dir(&self, app: &AppHandle) -> std::path::PathBuf {
        match &self.output_dir {
            Some(dir) => dir.into(),
            None => app
                .path()
                .document_dir()
                .unwrap_or_else(|_| std::env::temp_dir())
                .join("MeetScribe"),
        }
    }
}

fn store_path() -> std::path::PathBuf {
    crate::portable::store_path(STORE_PATH)
}

pub fn load(app: &AppHandle) -> MeetingSettings {
    let Ok(store) = app.store(store_path()) else {
        return MeetingSettings::default();
    };
    store
        .get(STORE_KEY)
        .and_then(|v| serde_json::from_value::<MeetingSettings>(v).ok())
        .unwrap_or_default()
        .sanitized()
}

pub fn save(app: &AppHandle, settings: &MeetingSettings) -> Result<(), String> {
    let store = app.store(store_path()).map_err(|e| e.to_string())?;
    store.set(
        STORE_KEY,
        serde_json::to_value(settings).map_err(|e| e.to_string())?,
    );
    store.save().map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn get_meeting_settings(app: AppHandle) -> MeetingSettings {
    load(&app)
}

#[tauri::command]
#[specta::specta]
pub fn set_meeting_settings(
    app: AppHandle,
    settings: MeetingSettings,
) -> Result<MeetingSettings, String> {
    let settings = settings.sanitized();
    save(&app, &settings)?;
    Ok(settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_clamps_and_clears_blanks() {
        let s = MeetingSettings {
            max_speakers: 12,
            output_dir: Some("  ".into()),
            asr_model: Some(String::new()),
            ..Default::default()
        }
        .sanitized();
        assert_eq!(s.max_speakers, 8);
        assert_eq!(s.output_dir, None);
        assert_eq!(s.asr_model, None);
        assert_eq!(
            MeetingSettings {
                max_speakers: 0,
                ..Default::default()
            }
            .sanitized()
            .max_speakers,
            1
        );
    }

    #[test]
    fn missing_fields_take_defaults() {
        let s: MeetingSettings = serde_json::from_str(r#"{"max_speakers":3}"#).unwrap();
        assert_eq!(s.max_speakers, 3);
        assert_eq!(s.latency_profile, LatencyProfile::Low);
        assert!(!s.save_audio);
    }
}
