# MeetScribe – status (2026-10-26)

Branch `meeting-mode`, local only (nothing pushed). Details per phase: `PLAN.md`,
`ARCHITECTURE-NOTES.md`, `SPIKE-RESULTS.md`.

| Gate                    | State                          | Evidence                                                                                                                    |
| ----------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 0 – fork, build, recon  | ✅                             | dev build runs under `com.wouter.meetscribe`; architecture notes                                                            |
| Rebrand                 | ✅ (placeholder art)           | name/icons/updater/NOTICE; 308 upstream tests still pass                                                                    |
| 1 – Python reference    | ✅                             | real time ≥ 0.32 s profiles on CPU+MPS; licence OpenMDW 1.1; Dutch OK                                                       |
| 2 – native route        | ⚠️ **needs you**               | NeMo-Speech.cpp chosen over ONNX; agent not permitted to build external C++ – run `scripts/meetscribe/build_nemo_speech.sh` |
| 3 – Rust diarizer crate | 🟡 code done, unverified       | `crates/diarize` + `diarize-cli`; FFI type-checks; parity/RTF pending the SDK build                                         |
| 4 – meeting mode in app | ✅ prototype (Python diarizer) | end-to-end on recorded meetings; live-mic demo with 3 people still to do                                                    |

## Gate 4 measurements (simulated input, Python sidecar diarizer, `low_latency`, Parakeet TDT 0.6B v3, M5)

Input fed through the real pipeline in real time via `MEETSCRIBE_SIMULATE_WAV` (replaces only the
microphone). Scored with `research/score_meeting.py` (word midpoint vs reference speaker,
best one-to-one mapping; words in reference gaps/overlap skipped).

| Recording                                                | Length   | Speakers found | Turns | Words scored | **Word speaker accuracy** |
| -------------------------------------------------------- | -------- | -------------- | ----- | ------------ | ------------------------- |
| Synthetic Dutch (MLS)                                    | 4.0 min  | 4/4            | 48    | 530          | **100 %**                 |
| AMI IS1009a (real meeting, far-field table mic, English) | 14.0 min | 4/4            | 144   | 1413         | **98.7 %**                |

Latency, AMI run (67 utterances): utterance end → text on screen **median 512 ms, p95 1.25 s,
max 4.2 s**. The segmenter waits 0.6 s of silence before closing an utterance (0.2 s of it is kept
as tail), so perceived "stopped talking → text" is about +0.4 s on top. Speaker labels settle
~1 s behind the audio (diarizer buffer).

Resources over the 14 min (sampled every 30 s; models run on the Metal GPU so process CPU stays
low): app 120–190 MB RSS, 1–8 % CPU; ASR worker 135–870 MB; Python diarizer 350–740 MB, 1–8 % CPU.
No growth over time. The native backend should remove the ~0.5 GB Python process.

Not yet measured: a **live 10+ minute test with 3 people in a room** (needs you), and the native
diarizer's speed/parity (needs the SDK build).

## How to run

```bash
# dev app (own data dir, updater off)
bun install
bun run tauri dev
# start/stop a meeting: ⌥⇧M, the tray item, the Meetings settings page, or
src-tauri/target/debug/handy --toggle-meeting
```

Until the native diarizer is built, meeting mode uses the Python sidecar, which needs the research
env once: `cd research && uv sync`. Transcripts go to `~/Documents/MeetScribe/` (configurable).

Native diarizer (once):

```bash
scripts/meetscribe/build_nemo_speech.sh      # builds + installs NeMo-Speech.cpp SDK into research/vendor/
cd src-tauri/crates/diarize && cargo build --release --bin diarize-cli
../../target/release/diarize-cli ../../../research/data/synthetic/nl_mix_4spk_seed0.wav \
  --model ~/.cache/huggingface/hub/models--nvidia--Nemotron-3-Diarization/snapshots/*/Nemotron-3-Diarization.q8_0.gguf \
  --dump-probs /tmp/native.f32 > /tmp/native.rttm
```

Then rebuild the app; the "Automatic" diarizer setting switches to native by itself.

## Known gaps / next steps

1. **Native backend verification** (after the SDK build): parity vs `research/out/streaming/*`
   (`research/compare_native.py`), RTF, and an app run with `diarizer_backend = native`. Packaging
   the NeMo dylibs into the `.app` bundle is not done yet (dev builds use an rpath).
2. **Live microphone demo** with 3 people (Gate 4 checklist).
3. Capture-ring overruns drop samples without telling the sink, which would shift the timeline for
   both workers equally; logged, not compensated.
4. Dictation is paused while a meeting runs (one microphone owner).
5. Other locales show English text for the new meeting strings (only en/nl translated).
6. Placeholder artwork; final name/branding still open.
7. Phase 5 items (not started): offline re-diarization at the end with the 30.4 s profile, LLM
   summary hook, speaker enrollment, multitalker ASR, signed build.
