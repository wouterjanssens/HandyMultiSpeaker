//! Links NeMo-Speech.cpp's C ABI when its SDK is installed.
//!
//! The SDK prefix is `NEMO_SPEECH_PREFIX`, defaulting to `research/vendor/nemo-speech-sdk` (what
//! `scripts/meetscribe/build_nemo_speech.sh` installs). Without it the crate still builds and
//! `NativeDiarizer::new` returns an error, so the app never depends on the SDK to compile.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nemo_speech)");
    println!("cargo::rerun-if-env-changed=NEMO_SPEECH_PREFIX");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let prefix = std::env::var_os("NEMO_SPEECH_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../research/vendor/nemo-speech-sdk"));
    let lib_dir = prefix.join("lib");
    let header = prefix.join("include/nemo_speech/diar.h");
    println!("cargo::rerun-if-changed={}", header.display());

    let lib = ["libnemo_speech_asr_c.dylib", "libnemo_speech_asr_c.so"]
        .iter()
        .any(|f| lib_dir.join(f).is_file());
    if !header.is_file() || !lib {
        println!(
            "cargo::warning=NeMo-Speech.cpp SDK not found at {} - native diarizer disabled \
             (run scripts/meetscribe/build_nemo_speech.sh)",
            prefix.display()
        );
        return;
    }
    let lib_dir = lib_dir.canonicalize().unwrap_or(lib_dir);
    println!("cargo::rustc-cfg=nemo_speech");
    println!("cargo::rustc-link-search=native={}", lib_dir.display());
    println!("cargo::rustc-link-lib=dylib=nemo_speech_asr_c");
    // dev builds find the dylibs in place; bundling them is a packaging step (Phase 5)
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
