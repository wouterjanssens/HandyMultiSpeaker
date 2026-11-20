//! Settings of the OpenAI cloud model, in their own store file so upstream `AppSettings` stays
//! untouched. Whether OpenAI is *used* is not in here: that is the ordinary model selection
//! (`selected_model == "openai"` for dictation, the meeting "ASR model" for meetings).

use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

const STORE_PATH: &str = "openai_settings.json";
const STORE_KEY: &str = "openai";
/// Where these settings lived before 2026-11-18, next to the meeting settings.
const LEGACY_STORE_PATH: &str = "meeting_settings.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct OpenAiSettings {
    /// Model behind the "OpenAI" entry: dictation, files and meeting text.
    pub transcribe_model: String,
    /// Model that labels speakers in meetings.
    pub diarize_model: String,
    /// Languages people speak (ISO 639-1), used when the language choice is "auto".
    pub languages: Vec<String>,
    /// Meetings: after a meeting, re-label speakers from the whole recording (more accurate
    /// than live).
    pub final_pass: bool,
    /// Meetings: write AI meeting notes (summary, decisions, action items) next to the
    /// transcript.
    pub notes: bool,
    pub notes_model: String,
}

impl Default for OpenAiSettings {
    fn default() -> Self {
        Self {
            transcribe_model: "gpt-transcribe".into(),
            diarize_model: "gpt-4o-transcribe-diarize".into(),
            final_pass: true,
            languages: vec!["nl".into(), "en".into()],
            notes: true,
            notes_model: "gpt-5.5".into(),
        }
    }
}

impl OpenAiSettings {
    fn sanitized(mut self) -> Self {
        let d = Self::default();
        let fix = |v: &mut String, def: String| {
            *v = v.trim().to_string();
            if v.is_empty() {
                *v = def;
            }
        };
        fix(&mut self.transcribe_model, d.transcribe_model);
        fix(&mut self.diarize_model, d.diarize_model);
        fix(&mut self.notes_model, d.notes_model);
        let mut langs: Vec<String> = Vec::new();
        for l in self.languages.iter().map(|l| l.trim().to_ascii_lowercase()) {
            let valid = (2..=3).contains(&l.len()) && l.chars().all(|c| c.is_ascii_lowercase());
            if valid && !langs.contains(&l) {
                langs.push(l);
            }
        }
        self.languages = if langs.is_empty() { d.languages } else { langs };
        self
    }
}

pub fn load(app: &AppHandle) -> OpenAiSettings {
    let Ok(store) = app.store(crate::portable::store_path(STORE_PATH)) else {
        return OpenAiSettings::default();
    };
    store
        .get(STORE_KEY)
        .and_then(|v| serde_json::from_value::<OpenAiSettings>(v).ok())
        .unwrap_or_default()
        .sanitized()
}

/// Moves settings saved by the first version (17 Nov 2026) out of `meeting_settings.json`. Its
/// `dictation: true` meant "OpenAI is the dictation model"; that becomes the model selection.
/// Runs once at startup, before the model manager checks the selection.
pub fn migrate_legacy(app: &AppHandle) {
    let Ok(old) = app.store(crate::portable::store_path(LEGACY_STORE_PATH)) else {
        return;
    };
    let Some(value) = old.get(STORE_KEY) else {
        return;
    };
    let dictation = value
        .get("dictation")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let settings = serde_json::from_value::<OpenAiSettings>(value)
        .unwrap_or_default()
        .sanitized();
    if let Err(e) = save(app, &settings) {
        log::warn!("Could not move the OpenAI settings: {e}");
        return;
    }
    if dictation {
        let mut app_settings = crate::settings::get_settings(app);
        app_settings.selected_model = super::MODEL_ID.to_string();
        crate::settings::write_settings(app, app_settings);
    }
    old.delete(STORE_KEY);
    let _ = old.save();
    log::info!("Moved the OpenAI settings to {STORE_PATH}");
}

pub fn save(app: &AppHandle, settings: &OpenAiSettings) -> Result<(), String> {
    let store = app
        .store(crate::portable::store_path(STORE_PATH))
        .map_err(|e| e.to_string())?;
    store.set(
        STORE_KEY,
        serde_json::to_value(settings).map_err(|e| e.to_string())?,
    );
    store.save().map_err(|e| e.to_string())
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct OpenAiKeyStatus {
    pub found: bool,
    /// Where the key came from: "environment", "keyring" or "post-processing settings".
    pub source: Option<String>,
}

#[tauri::command]
#[specta::specta]
pub fn get_openai_settings(app: AppHandle) -> OpenAiSettings {
    load(&app)
}

#[tauri::command]
#[specta::specta]
pub fn set_openai_settings(
    app: AppHandle,
    settings: OpenAiSettings,
) -> Result<OpenAiSettings, String> {
    let settings = settings.sanitized();
    save(&app, &settings)?;
    Ok(settings)
}

#[tauri::command]
#[specta::specta]
pub fn openai_key_status(app: AppHandle) -> OpenAiKeyStatus {
    match super::client::api_key(&app) {
        Some((_, source)) => OpenAiKeyStatus {
            found: true,
            source: Some(source.to_string()),
        },
        None => OpenAiKeyStatus {
            found: false,
            source: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_keeps_valid_languages_and_restores_blank_models() {
        let s = OpenAiSettings {
            transcribe_model: "  ".into(),
            languages: vec!["NL".into(), "en".into(), "nl".into(), "x-y".into()],
            ..Default::default()
        }
        .sanitized();
        assert_eq!(s.transcribe_model, "gpt-transcribe");
        assert_eq!(s.languages, vec!["nl", "en"]);
        let s = OpenAiSettings {
            languages: vec![],
            ..Default::default()
        }
        .sanitized();
        assert_eq!(s.languages, vec!["nl", "en"]);
    }
}
