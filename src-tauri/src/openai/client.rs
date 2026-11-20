//! Minimal blocking OpenAI client: file transcription, diarized transcription, Responses API.
//!
//! Call only from plain OS threads (see [`super::off_runtime`]).

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use reqwest::blocking::{multipart, Client as Http, Response};
use serde::Deserialize;
use tauri::AppHandle;

const BASE_URL: &str = "https://api.openai.com/v1";
const SAMPLE_RATE: u32 = 16_000;

/// The API key and where it was found. Never logged.
pub fn api_key(app: &AppHandle) -> Option<(String, &'static str)> {
    if let Ok(k) = std::env::var("OPENAI_API_KEY") {
        if !k.trim().is_empty() {
            return Some((k.trim().to_string(), "environment"));
        }
    }
    if let Some(k) = keyring_key() {
        return Some((k, "keyring"));
    }
    crate::settings::get_settings(app)
        .post_process_api_keys
        .get("openai")
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .map(|k| (k, "post-processing settings"))
}

/// `secret-tool lookup service openai account api-key` (Linux) or the macOS keychain item
/// `openai` / `api-key`.
fn keyring_key() -> Option<String> {
    #[cfg(target_os = "linux")]
    let out = std::process::Command::new("secret-tool")
        .args(["lookup", "service", "openai", "account", "api-key"])
        .output()
        .ok()?;
    #[cfg(target_os = "macos")]
    let out = std::process::Command::new("security")
        .args([
            "find-generic-password",
            "-s",
            "openai",
            "-a",
            "api-key",
            "-w",
        ])
        .output()
        .ok()?;
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return None;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let k = String::from_utf8(out.stdout).ok()?.trim().to_string();
        (out.status.success() && !k.is_empty()).then_some(k)
    }
}

/// 16 kHz mono 16-bit WAV in memory.
pub fn wav_bytes(pcm: &[f32]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + pcm.len() * 2));
    {
        let mut w = hound::WavWriter::new(&mut cursor, spec).expect("in-memory wav");
        for &x in pcm {
            let _ = w.write_sample((x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
        }
        let _ = w.finalize();
    }
    cursor.into_inner()
}

pub fn wav_data_url(pcm: &[f32]) -> String {
    format!(
        "data:audio/wav;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(wav_bytes(pcm))
    )
}

/// A keyword the API accepts (one line, no angle brackets).
fn valid_keyword(k: &str) -> bool {
    !k.trim().is_empty() && !k.contains(['<', '>', '\r', '\n'])
}

/// One speaker segment from `diarized_json`, times in seconds relative to the request audio.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DiarSegment {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub text: String,
}

/// A voice the diarizer should recognise: a name and a 2–10 s reference clip.
#[derive(Debug, Clone)]
pub struct KnownSpeaker {
    pub name: String,
    pub data_url: String,
}

#[derive(Clone)]
pub struct Client {
    http: Http,
    key: String,
}

impl Client {
    pub fn from_app(app: &AppHandle) -> Result<Self> {
        let (key, _) = api_key(app).ok_or_else(|| {
            anyhow!("No OpenAI API key: set OPENAI_API_KEY, store it in the keyring (service=openai account=api-key) or enter it under Post Processing → OpenAI")
        })?;
        Ok(Self::new(key))
    }

    pub fn new(key: String) -> Self {
        // building the client on a plain thread; it owns no runtime until used
        let http = Http::builder()
            .timeout(Duration::from_secs(180))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .expect("http client");
        Self { http, key }
    }

    /// Sends a request built by `make`, up to three tries in all: rate limits, server errors and
    /// network failures are tried again after a short pause (0.8 s, then 1.6 s).
    fn send(
        &self,
        what: &str,
        make: impl Fn() -> reqwest::blocking::RequestBuilder,
    ) -> Result<Response> {
        let mut last_err = None;
        for attempt in 0..3 {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(800 * attempt));
            }
            match make().bearer_auth(&self.key).send() {
                Ok(r) if r.status().is_success() => return Ok(r),
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().unwrap_or_default();
                    let msg = serde_json::from_str::<serde_json::Value>(&body)
                        .ok()
                        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                        .unwrap_or(body);
                    let err = anyhow!("OpenAI {what} failed ({status}): {msg}");
                    if status.as_u16() == 429 || status.is_server_error() {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
                Err(e) => last_err = Some(anyhow!("OpenAI {what} request failed: {e}")),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("OpenAI {what} failed")))
    }

    fn audio_part(pcm: &[f32]) -> Result<multipart::Part> {
        multipart::Part::bytes(wav_bytes(pcm))
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .context("wav part")
    }

    /// Plain text of `pcm` (`gpt-transcribe`: `languages[]` and `keywords[]` supported).
    pub fn transcribe(
        &self,
        model: &str,
        pcm: &[f32],
        languages: &[String],
        keywords: &[String],
    ) -> Result<String> {
        #[derive(Deserialize)]
        struct Out {
            text: String,
        }
        let new_api = model.starts_with("gpt-transcribe");
        let resp = self.send("transcription", || {
            let mut form = multipart::Form::new()
                .text("model", model.to_string())
                .text("response_format", "json")
                .part("file", Self::audio_part(pcm).expect("wav"));
            if new_api {
                for l in languages {
                    form = form.text("languages[]", l.clone());
                }
                for k in keywords.iter().filter(|k| valid_keyword(k)).take(100) {
                    form = form.text("keywords[]", k.trim().to_string());
                }
            } else if let [one] = languages {
                form = form.text("language", one.clone());
            }
            self.http
                .post(format!("{BASE_URL}/audio/transcriptions"))
                .multipart(form)
        })?;
        Ok(resp.json::<Out>().context("transcription response")?.text)
    }

    /// Speaker segments of `pcm`. Known speakers come back under their names; others as `A`,
    /// `B`, … (fresh per request).
    pub fn diarize(
        &self,
        model: &str,
        pcm: &[f32],
        language: Option<&str>,
        known: &[KnownSpeaker],
    ) -> Result<Vec<DiarSegment>> {
        #[derive(Deserialize)]
        struct Out {
            #[serde(default)]
            segments: Vec<DiarSegment>,
        }
        let resp = self.send("diarization", || {
            let mut form = multipart::Form::new()
                .text("model", model.to_string())
                .text("response_format", "diarized_json")
                .text("chunking_strategy", "auto")
                .part("file", Self::audio_part(pcm).expect("wav"));
            if let Some(l) = language {
                form = form.text("language", l.to_string());
            }
            for k in known.iter().take(4) {
                form = form
                    .text("known_speaker_names[]", k.name.clone())
                    .text("known_speaker_references[]", k.data_url.clone());
            }
            self.http
                .post(format!("{BASE_URL}/audio/transcriptions"))
                .multipart(form)
        })?;
        Ok(resp.json::<Out>().context("diarization response")?.segments)
    }

    /// One Responses API turn; returns the output text.
    pub fn respond(&self, model: &str, instructions: &str, input: &str) -> Result<String> {
        let mut body = serde_json::json!({
            "model": model,
            "instructions": instructions,
            "input": input,
        });
        if model.starts_with("gpt-5") || model.starts_with('o') {
            body["reasoning"] = serde_json::json!({ "effort": "medium" });
        }
        let resp = self.send("notes", || {
            self.http.post(format!("{BASE_URL}/responses")).json(&body)
        })?;
        let v: serde_json::Value = resp.json().context("responses body")?;
        let text: String = v["output"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|o| o["type"] == "message")
            .flat_map(|o| o["content"].as_array().cloned().unwrap_or_default())
            .filter(|c| c["type"] == "output_text")
            .filter_map(|c| c["text"].as_str().map(str::to_string))
            .collect();
        if text.trim().is_empty() {
            bail!("OpenAI returned no text (status {})", v["status"]);
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_is_16k_mono_pcm16() {
        let bytes = wav_bytes(&[0.0, 0.5, -1.0, 2.0]);
        let r = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(r.spec().sample_rate, 16_000);
        assert_eq!(r.spec().channels, 1);
        let s: Vec<i16> = r.into_samples().map(|s| s.unwrap()).collect();
        assert_eq!(s, vec![0, 16383, -32767, 32767]);
    }

    #[test]
    fn keywords_with_markup_or_newlines_are_dropped() {
        assert!(valid_keyword("Go4TMS"));
        assert!(!valid_keyword("a<b"));
        assert!(!valid_keyword("two\nlines"));
        assert!(!valid_keyword("  "));
    }
}
