//! Spoken-language hints for meeting transcription.
//!
//! Whether a hint does anything depends on the model, which the app's generic "multilingual"
//! flag doesn't capture:
//! - Whisper: always honours it.
//! - Parakeet family: only the prompt-conditioned variants (e.g. Nemotron 3.5 ASR Streaming),
//!   recognisable by a `stt.parakeet.prompt.dictionary.locales` table; Parakeet TDT v3 has none
//!   and always auto-detects.
//! - Others: unknown; the hint is passed if the model lists the language.
//!
//! Models also differ in codes (`nl` vs `nl-NL`), so a short code is mapped to one the model
//! accepts – an unknown code would fail the whole utterance.

use std::path::Path;

const KEY_ARCH: &str = "general.architecture";
const KEY_LANGUAGES: &str = "general.languages";
const KEY_PROMPT_LOCALES: &str = "stt.parakeet.prompt.dictionary.locales";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HintSupport {
    /// `Some(false)`: the model ignores hints; `None`: unknown.
    pub honored: Option<bool>,
    /// Codes the model accepts, most specific source first.
    pub codes: Vec<String>,
}

impl HintSupport {
    pub fn from_metadata(
        arch: Option<&str>,
        languages: Vec<String>,
        prompt_locales: Option<Vec<String>>,
    ) -> Self {
        match (arch, prompt_locales) {
            (Some("whisper"), _) => Self {
                honored: Some(true),
                codes: languages,
            },
            // transcribe-cpp validates the hint against `general.languages` before the
            // prompt dictionary sees it, so only those codes (e.g. "nl-NL") are usable
            (Some("parakeet"), Some(locales)) if !locales.is_empty() => Self {
                honored: Some(true),
                codes: if languages.is_empty() {
                    locales
                } else {
                    languages
                },
            },
            (Some("parakeet"), _) => Self {
                honored: Some(false),
                codes: Vec::new(),
            },
            _ => Self {
                honored: None,
                codes: languages,
            },
        }
    }

    /// Reads the model's GGUF header; unreadable files give "unknown".
    pub fn probe(path: &Path) -> Self {
        let keys = [KEY_ARCH, KEY_LANGUAGES, KEY_PROMPT_LOCALES];
        match crate::managers::model_capabilities::read_header_metadata(path, &keys) {
            Ok(meta) => Self::from_metadata(
                meta.get_str(KEY_ARCH),
                meta.get_string_array(KEY_LANGUAGES).unwrap_or_default(),
                meta.get_string_array(KEY_PROMPT_LOCALES),
            ),
            Err(_) => Self::default(),
        }
    }

    /// The code to pass for `language` ("auto" or ISO 639-1), or `None` to auto-detect.
    pub fn resolve(&self, language: &str) -> Option<String> {
        if language == "auto" || self.honored == Some(false) {
            return None;
        }
        if self.codes.iter().any(|c| c == language) {
            return Some(language.to_string());
        }
        let prefix = format!("{language}-");
        self.codes.iter().find(|c| c.starts_with(&prefix)).cloned()
    }

    /// True when a non-automatic choice will have no effect with this model.
    pub fn ignores(&self, language: &str) -> bool {
        language != "auto" && self.resolve(language).is_none() && self.honored.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parakeet_tdt_without_prompt_ignores_hints() {
        let h = HintSupport::from_metadata(Some("parakeet"), s(&["nl", "en", "da"]), None);
        assert_eq!(h.honored, Some(false));
        assert_eq!(h.resolve("nl"), None);
        assert!(h.ignores("nl"));
        assert!(!h.ignores("auto"));
    }

    #[test]
    fn prompt_conditioned_parakeet_maps_short_codes() {
        let h = HintSupport::from_metadata(
            Some("parakeet"),
            s(&["en-US", "nl-NL", "da-DK"]),
            Some(s(&["en-US", "en", "nl-NL", "nl", "da-DK", "fr-FR", "fr"])),
        );
        // the declared regional codes win: the engine rejects anything else
        assert_eq!(h.resolve("nl").as_deref(), Some("nl-NL"));
        assert_eq!(h.resolve("da").as_deref(), Some("da-DK"));
        assert_eq!(h.resolve("fr"), None);
        assert_eq!(h.resolve("auto"), None);
        assert!(h.ignores("fr"));
    }

    #[test]
    fn whisper_honours_listed_languages() {
        let h = HintSupport::from_metadata(Some("whisper"), s(&["en", "nl", "fr"]), None);
        assert_eq!(h.resolve("nl").as_deref(), Some("nl"));
        assert_eq!(h.resolve("da"), None);
        assert!(h.ignores("da"));
    }

    /// `MEETSCRIBE_PROBE_GGUF=/path/model.gguf cargo test probe_real -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_real_model() {
        let path = std::env::var("MEETSCRIBE_PROBE_GGUF").expect("set MEETSCRIBE_PROBE_GGUF");
        let h = HintSupport::probe(Path::new(&path));
        println!(
            "{path}: honored={:?} nl={:?} da={:?} fr={:?} en={:?}",
            h.honored,
            h.resolve("nl"),
            h.resolve("da"),
            h.resolve("fr"),
            h.resolve("en")
        );
    }

    /// Does the hint change the output? `MEETSCRIBE_PROBE_GGUF=model.gguf
    /// MEETSCRIBE_PROBE_WAV=16k-mono.wav cargo test hint_effect -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn hint_effect_on_real_model() {
        use transcribe_cpp::{Model, RunOptions};
        let model_path = std::env::var("MEETSCRIBE_PROBE_GGUF").expect("set MEETSCRIBE_PROBE_GGUF");
        let wav = std::env::var("MEETSCRIBE_PROBE_WAV").expect("set MEETSCRIBE_PROBE_WAV");
        let pcm: Vec<f32> = hound::WavReader::open(&wav)
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let hint = HintSupport::probe(Path::new(&model_path));
        let model = Model::load(&model_path).unwrap();
        let mut session = model.session().unwrap();
        for lang in ["auto", "nl", "en"] {
            let t = session
                .run(
                    &pcm,
                    &RunOptions {
                        language: hint.resolve(lang),
                        ..Default::default()
                    },
                )
                .unwrap();
            println!(
                "[{lang} -> {:?}] detected={:?} words={} kind={:?}\n  {}",
                hint.resolve(lang),
                t.language,
                t.words.len(),
                t.timestamp_kind,
                t.text.trim()
            );
        }
    }

    #[test]
    fn unknown_models_pass_listed_codes_only_and_never_warn() {
        let h = HintSupport::from_metadata(Some("canary"), s(&["en", "fr"]), None);
        assert_eq!(h.resolve("fr").as_deref(), Some("fr"));
        assert_eq!(h.resolve("nl"), None);
        assert!(!h.ignores("nl"));
    }
}
