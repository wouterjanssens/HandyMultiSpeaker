//! The Nemotron 3 Diarization weights for the native backend: one pinned GGUF from the model's
//! Hugging Face repo (OpenMDW 1.1), fetched into the shared HF cache on first use and verified by
//! SHA-256 every time it is opened. Never bundled or committed.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};

const REPO: &str = "nvidia/Nemotron-3-Diarization";
/// Revision the file below was verified against (2026-09-24 upload).
const REVISION: &str = "f667ed73aee57d40cc39428eb768b4fd87a0a29e";
const FILE: &str = "Nemotron-3-Diarization.q8_0.gguf";
const SHA256: &str = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1";

fn repo() -> Repo {
    Repo::with_revision(REPO.to_string(), RepoType::Model, REVISION.to_string())
}

fn sha256_hex(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn verified(path: PathBuf) -> Result<PathBuf> {
    let actual = sha256_hex(&path).with_context(|| format!("cannot read {}", path.display()))?;
    if actual != SHA256 {
        bail!(
            "diarizer model checksum mismatch at {} (expected {SHA256}, got {actual}); delete it to re-download",
            path.display()
        );
    }
    Ok(path)
}

/// Path to the verified diarizer GGUF, downloading it (~107 MB) if it isn't cached yet.
/// Blocks; call from a worker thread.
pub fn ensure_diarizer_model(app: &AppHandle) -> Result<PathBuf> {
    if let Some(path) = Cache::from_env().repo(repo()).get(FILE) {
        return verified(path);
    }
    log::info!("Downloading {REPO}/{FILE} @ {REVISION}");
    let _ = app.emit("meeting:model_download", FILE);
    let path = tauri::async_runtime::block_on(async {
        ApiBuilder::from_env()
            .with_token(None)
            .with_progress(false)
            .build()?
            .repo(repo())
            .get(FILE)
            .await
    })
    .context("failed to download the diarizer model")?;
    verified(path)
}
