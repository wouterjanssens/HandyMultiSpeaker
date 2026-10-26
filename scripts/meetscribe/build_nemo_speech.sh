#!/usr/bin/env bash
# Builds NVIDIA NeMo-Speech.cpp (Apache-2.0) – standalone diarization with Metal – and installs
# its SDK (C headers + libraries) into research/vendor/nemo-speech-sdk, where the `diarize`
# crate links against it. macOS / Apple Silicon only.
#
# Run it yourself from the repo root:
#   scripts/meetscribe/build_nemo_speech.sh
#
# Then smoke-test the CLI on the Dutch test file:
#   research/vendor/nemo-speech-sdk/bin/nemo-speech diarize \
#     research/data/synthetic/nl_mix_4spk_seed0.wav \
#     --diar-model ~/.cache/huggingface/hub/models--nvidia--Nemotron-3-Diarization/snapshots/*/Nemotron-3-Diarization.q8_0.gguf \
#     --format rttm
set -euo pipefail

VERSION="${NEMO_SPEECH_VERSION:-v0.2.0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/research/vendor/NeMo-Speech.cpp"
PREFIX="$ROOT/research/vendor/nemo-speech-sdk"
PRESET="metal-diar"

command -v cmake >/dev/null || { echo "cmake missing: brew install cmake"; exit 1; }
command -v ninja >/dev/null || { echo "ninja missing: brew install ninja"; exit 1; }

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$VERSION" https://github.com/NVIDIA/NeMo-Speech.cpp.git "$SRC"
fi
cd "$SRC"
git submodule update --init --depth 1 llama.cpp

scripts/configure.sh "$PRESET" -DNEMO_SPEECH_BUILD_EXAMPLES=ON
cmake --build --preset "$PRESET"
cmake --install "build/$PRESET" --prefix "$PREFIX"

echo
echo "Installed NeMo-Speech.cpp $VERSION ($PRESET) into $PREFIX"
ls "$PREFIX"/include/nemo_speech "$PREFIX"/lib | sed 's/^/  /'
