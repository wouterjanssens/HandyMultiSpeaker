# MeetScribe – plan log

## 2026-10-26 – Phase 0: fork, build, recon

Plan:

1. Install missing tooling (rustup, bun, uv, cmake via Homebrew).
2. Use the existing fork `wouterjanssens/HandyMultiSpeaker` (origin) + `cjpais/Handy` (upstream);
   branch `meeting-mode`.
3. Build unmodified; run the dev build under our own bundle id so the installed Handy's data is
   never touched.
4. Recon → `ARCHITECTURE-NOTES.md`.

Outcome: done. Build OK. Key findings: upstream now uses `handy-recorder` + `transcribe-cpp` for
everything; transcribe-cpp has a (4-speaker, one-shot) Sortformer diarizer; NVIDIA's NeMo-Speech.cpp
has a native, Metal-capable, streaming C API for Nemotron 3 Diarization. See notes §5–6.

Open questions for Gate 0: project name; whether to replace the ONNX route (Phase 2) with
NeMo-Speech.cpp.
