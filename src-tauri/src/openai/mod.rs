//! meetscribe: OpenAI as a transcription model.
//!
//! "OpenAI" is an entry in the ordinary model list ([`MODEL_ID`], [`model_info`], source
//! `Cloud`): selecting it for dictation or as the meeting model is what turns it on, and nothing
//! is downloaded or loaded for it. Behind it:
//! - text (dictation, files, meetings): `gpt-transcribe` (language list + keywords);
//! - meeting speakers: `gpt-4o-transcribe-diarize` per utterance, kept consistent across the
//!   meeting with up to four enrolled voice references ([`speakers`]);
//! - meeting notes after a meeting: the Responses API ([`notes`]).
//!
//! Audio only leaves the machine while OpenAI is the selected model. The API key is read from
//! `OPENAI_API_KEY`, the OS keyring (`service=openai account=api-key`) or the post-processing
//! "OpenAI" key, in that order, and is never written anywhere.

pub mod client;
pub mod notes;
pub mod settings;
pub mod speakers;

pub use client::Client;
pub use settings::OpenAiSettings;

/// Runs `f` on a plain OS thread. The blocking HTTP client must not run (or be dropped) on a
/// tokio worker, and some callers (the dictation stop path) are async tasks.
pub fn off_runtime<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::spawn(f)
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e))
}

/// Id of the OpenAI entry in the model list (also a valid meeting "ASR model").
pub const MODEL_ID: &str = "openai";

/// Languages `gpt-transcribe` handles (OpenAI's supported-language list, ISO 639-1).
const LANGUAGES: &[&str] = &[
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da", "nl", "en", "et", "fi",
    "fr", "gl", "de", "el", "he", "hi", "hu", "is", "id", "it", "ja", "kn", "kk", "ko", "lv", "lt",
    "mk", "ms", "mr", "mi", "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw",
    "sv", "tl", "ta", "th", "tr", "uk", "ur", "vi", "cy",
];

/// The OpenAI entry for the model list.
pub fn model_info() -> crate::managers::model::ModelInfo {
    crate::managers::model::ModelInfo {
        id: MODEL_ID.to_string(),
        name: "OpenAI".to_string(),
        description:
            "Cloud transcription and meeting speakers. Audio is sent to OpenAI; needs an API key."
                .to_string(),
        filename: String::new(),
        source: crate::managers::model::ModelSource::Cloud,
        size_mb: 0,
        is_downloaded: true,
        is_downloading: false,
        partial_size: 0,
        accuracy_score: 0.95,
        speed_score: 0.7,
        supports_translation: false,
        is_recommended: false,
        supported_languages: LANGUAGES.iter().map(|l| l.to_string()).collect(),
        supports_language_selection: true,
        is_custom: false,
        supports_streaming: false,
        supports_language_detection: true,
    }
}

/// True when `model_id` is the OpenAI entry.
pub fn is_model(model_id: &str) -> bool {
    model_id == MODEL_ID
}

/// True when OpenAI is the selected dictation model.
pub fn is_selected(app: &tauri::AppHandle) -> bool {
    is_model(&crate::settings::get_settings(app).selected_model)
}

/// True when meetings run on OpenAI: chosen as the meeting model, or the meeting follows the
/// dictation model and that is OpenAI.
pub fn meetings_enabled(app: &tauri::AppHandle) -> bool {
    match crate::meeting::settings::load(app).asr_model.as_deref() {
        Some(id) => is_model(id),
        None => is_selected(app),
    }
}

/// One `gpt-transcribe` request for a clip (dictation, or one chunk of a file).
pub fn transcribe(app: &tauri::AppHandle, samples: Vec<f32>) -> anyhow::Result<String> {
    if samples.is_empty() {
        return Ok(String::new());
    }
    let s = settings::load(app);
    let app_settings = crate::settings::get_settings(app);
    let languages = match app_settings.selected_language.as_str() {
        "auto" | "" => s.languages.clone(),
        code => vec![code
            .split(['-', '_'])
            .next()
            .unwrap_or(code)
            .to_ascii_lowercase()],
    };
    let keywords = app_settings.custom_words.clone();
    let client = Client::from_app(app)?;
    let started = std::time::Instant::now();
    let model = s.transcribe_model.clone();
    let secs = samples.len() as f64 / 16_000.0;
    let text = off_runtime(move || client.transcribe(&model, &samples, &languages, &keywords))?;
    log::info!(
        "OpenAI transcription: {:.1}s of audio in {:.2}s",
        secs,
        started.elapsed().as_secs_f64()
    );
    Ok(text.trim().to_string())
}
