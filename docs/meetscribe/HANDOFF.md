# Handoff: Live multi-speaker fork of Handy (Nemotron 3 Diarization)

You are Claude Code, working in an empty folder created for this project. Your job is to fork **cjpais/Handy**, set up the dev environment, and start building a **live, speaker-attributed meeting transcription mode** using **NVIDIA Nemotron 3 Diarization** as the streaming diarizer. This is the "hard path": speaker labels appear _during_ the meeting, not only after it.

Read this whole file before doing anything. Work in phases, commit at the end of each phase, and stop at every **🛑 GATE** to report to me before continuing.

---

## 1. Context

**Who I am:** Wouter, based in Belgium. I'm on an Apple Silicon Mac. I want this for physical meetings (one laptop or conference mic on the table, 2–6 people, often speaking Dutch/Flemish, sometimes English) so I can generate meeting notes with who-said-what.

**Handy today** (https://github.com/cjpais/Handy, MIT):

- Tauri v2 app. Rust backend, React + TypeScript + Tailwind frontend.
- Key crates/libs: `transcribe-cpp` (Whisper-family, GGML/GGUF), `transcribe-rs` (Parakeet), `cpal` (audio I/O), `vad-rs` (Silero VAD), `rdev` (shortcuts), `rubato` (resampling).
- Push-to-talk or toggle recording. Transcribes on release and pastes into the focused text field.
- CLI: `handy --toggle-transcription`, `--toggle-post-process`, `--cancel`, `--start-hidden`, `--no-tray`, `--debug`.
- All models now emit timestamps (upstream PR #1602). We need these for alignment.
- The maintainer has said diarization and meeting transcription are out of scope for upstream (Discussion #599). This fork will **never** be merged upstream. Don't plan for that, but keep our code isolated so upstream updates can still be merged _into_ the fork.
- Handy's own GGUF port of `nvidia/multitalker-parakeet-streaming-0.6b-v1` ships **single-speaker mode only**. The diarization and multi-instance machinery was not ported. Don't assume any existing diarization code exists in Handy or transcribe.cpp. Verify, but expect none.

**Nemotron 3 Diarization** (`nvidia/Nemotron-3-Diarization` on Hugging Face):

- Open-weight, ~100M params, released Sept 2026. Diarization only: it outputs _who spoke when_, not words.
- Up to **8 speakers**. Accuracy degrades above ~4, so make max-speakers configurable.
- Sortformer-style: speakers are ordered by first arrival. Streaming uses an **Arrival-Order Speaker Cache (AOSC) + FIFO queue**.
- Output: frame-level speaker-activity logits, shape `(1, num_frames, 8)`, one frame per **10 ms**.
- One checkpoint supports latency profiles from an **80 ms** input buffer to a **30.4 s** offline-style buffer.
- Available via 🤗 Transformers (`AutoModelForAudioFrameClassification` + `AutoProcessor`, streaming via a `speaker_cache` argument and `processor.streaming_latency_ms`) and NeMo (`.nemo` checkpoint).
- Officially targets NVIDIA GPUs. **There is no known Mac/Metal/CoreML path.** Creating one is the core technical risk of this project.

---

## 2. Goal and non-goals

**Goal (end state):** A "Meeting mode" in the fork:

1. Start or stop with a shortcut, a tray menu item, or the CLI (`--toggle-meeting`).
2. While recording, a meeting window shows a live, growing transcript grouped into turns: `Speaker 1: …`, `Speaker 2: …`.
3. Speakers can be renamed mid- or post-meeting (Speaker 2 → "Jan"), and the rename applies to the whole transcript.
4. On stop, save `YYYY-MM-DD_HHMM_meeting.md` (and optionally a `.json` with word-level timestamps and speaker IDs) to a configurable folder, plus the raw audio (optional, off by default).
5. Everything runs **locally on an Apple Silicon Mac**. No audio leaves the machine.

**Non-goals for now:** Upstream contribution, voice enrollment/recognition across meetings, LLM summarisation, Windows/Linux packaging (keep them compiling if cheap, but Mac is the target), overlapping-speech transcription via multitalker Parakeet (possible later phase).

---

## 3. Hard constraints

- **Branding:** Handy's README states the name, logo, icon and brand assets are _not_ open source, and forks must use their own branding without implying endorsement. Before Phase 1, ask me for the project name. Until I answer, use the working name `MeetScribe` and the bundle identifier `com.wouter.meetscribe`. Replace all user-visible "Handy" names, icons and the bundle id (`com.pais.handy`) in the fork. Keep the upstream MIT license and copyright notice, and add a NOTICE line saying it's a fork of cjpais/Handy.
- **Data dir:** use our own bundle id so the fork never touches an installed Handy's settings or models.
- **Isolation:** put new Rust code in new modules (e.g. `src-tauri/src/meeting/`, a `diarize` crate). Keep edits to existing upstream files minimal and marked `// meetscribe:` so upstream merges stay easy.
- **Model licence:** check the Nemotron 3 Diarization model card licence before bundling or auto-downloading weights, and report what it says at Gate 1.
- **No secrets in the repo.** If a Hugging Face token is needed, read it from env (`HF_TOKEN`) and never commit it.
- **Don't push** to any remote or create GitHub releases without asking me.

---

## 4. Phase 0: Fork, clone, build, recon

1. Check tooling: `gh auth status`, `git`, Rust (`rustup`, stable), Bun (the repo uses `bun.lock`), Xcode command-line tools, Python 3.11+ with `uv` or `venv`. Install what's missing, or tell me if it needs my action.
2. Fork and clone into the current folder:
   ```bash
   gh repo fork cjpais/Handy --clone=true --remote=true
   cd Handy
   git remote -v   # origin = my fork, upstream = cjpais/Handy
   git checkout -b meeting-mode
   ```
   Renaming the GitHub repo to the final project name can happen later, after I confirm the name.
3. Follow `BUILD.md` and get an unmodified dev build running on the Mac (`bun install`, `bun run tauri dev` or whatever BUILD.md specifies). Confirm normal dictation works.
4. Read `CLAUDE.md`, `AGENTS.md` and `CONTRIBUTING.md` in the repo and follow their conventions.
5. **Recon report:** map and write down in `docs/meetscribe/ARCHITECTURE-NOTES.md`:
   - The audio capture path (cpal stream → resampling → VAD → buffer). Which thread owns it and how samples flow.
   - Where transcription is invoked, which engine abstraction exists, and how timestamps come back (segment-level or word-level? check both engines).
   - How recording state, shortcuts, the tray and CLI flags are wired, so we can add `--toggle-meeting`.
   - How settings are stored and exposed to the frontend (Tauri commands/events).
   - Whether `transcribe.cpp` exposes anything reusable for diarization (expect not).

🛑 **GATE 0:** Report build status, the architecture notes, and any surprises. Ask me for the project name.

---

## 5. Phase 1: Reference implementation in Python (ground truth)

Before touching Rust, prove the model works for my use case and produce reference outputs to test the port against.

1. In `research/` (excluded from the app build), set up a Python env with `transformers` (a version that includes Nemotron 3 Diarization, v5.18.0 or later), `torch`, `torchaudio`, `soundfile`.
2. Write `research/diarize_offline.py`: runs the model offline on a WAV and writes RTTM + a JSON of segments.
3. Write `research/diarize_streaming.py`: runs the model **in streaming mode** using the processor's streaming API and the `speaker_cache`, feeding fixed-size chunks the way a live mic would. Log `streaming_latency_ms`, per-chunk compute time on Mac CPU and on MPS (if PyTorch MPS works for this model), and the final segments. Save per-chunk logits as `.npy` so we have exact numerical references.
4. Test audio: ask me for a real meeting recording if I have one. Otherwise build a synthetic 3–4 speaker test by concatenating/overlapping public-domain speech clips (document sources), including one Dutch clip if one can be found.
5. Measure and record in `docs/meetscribe/SPIKE-RESULTS.md`:
   - Real-time factor at 2–3 latency profiles (e.g. ~80 ms, ~1 s, ~5 s).
   - Speaker-count accuracy and rough DER on the test file.
   - Exactly how the streaming state is structured: tensor names, shapes and dtypes of the speaker cache, FIFO queue and any other carried state, and what gets updated each step.

🛑 **GATE 1:** Report results, the licence, and your recommendation for the latency profile. **If the model can't run at least ~1× real-time on Mac CPU or MPS in Python, stop and tell me.** That changes the whole plan.

---

## 6. Phase 2: Export for native inference (main risk)

Goal: run the streaming step from Rust with no Python at runtime.

**Primary route: ONNX + `ort` crate.**

1. Export the model's **single streaming step** as an ONNX graph: inputs = audio features for one chunk + all carried state tensors; outputs = logits for the chunk + updated state tensors. Keep any non-tensor bookkeeping (e.g. AOSC selection logic, if it's implemented in Python control flow) **out of the graph** and plan to reimplement it in Rust. Document exactly what was split where.
2. Check whether the feature extraction (mel/fbank) is inside the model or in the processor. If it's in the processor, either export it separately or reimplement it in Rust, and verify bit-close parity on the feature level first.
3. Validate with `onnxruntime` in Python: run the exported step on the same chunks as Phase 1 and compare logits to the saved `.npy` references (target: max abs diff < 1e-3 on fp32; record the actual numbers).
4. Try fp16 and/or int8 dynamic quantisation, and record accuracy vs speed.
5. Check onnxruntime with the **CoreML execution provider** on the Mac. Record which ops fall back to CPU and the resulting speed.

**Fallback routes, only if ONNX export of the streaming step fails or is too slow.** Don't start these without asking me:

- **B:** Port the model to GGML inside our own fork of `transcribe.cpp` (big effort; Metal acceleration comes for free).
- **C:** A Python sidecar process (PyTorch MPS) that the Tauri app talks to over stdin/stdout or a local socket. Fastest to working demo, but breaks the single-binary story. Acceptable as a temporary prototype.

🛑 **GATE 2:** Report parity numbers, speed per execution provider, what had to be reimplemented outside the graph, and your recommended route.

---

## 7. Phase 3: Rust diarization crate

Create a standalone crate, e.g. `src-tauri/crates/diarize` (or a workspace member, following repo conventions), with no Tauri dependency, so it's testable on its own.

- API sketch (adjust to what Phase 2 found):
  ```rust
  pub struct StreamingDiarizer { /* ort session, state tensors, AOSC/FIFO bookkeeping, config */ }
  pub struct DiarizerConfig { max_speakers: u8, latency_profile: LatencyProfile, threshold: f32 }
  impl StreamingDiarizer {
      pub fn new(model_path: &Path, cfg: DiarizerConfig) -> Result<Self>;
      /// Feed 16 kHz mono f32 samples; returns finalized frames (10 ms each) with per-speaker probabilities.
      pub fn push_audio(&mut self, samples: &[f32]) -> Result<Vec<SpeakerFrame>>;
      pub fn flush(&mut self) -> Result<Vec<SpeakerFrame>>;
      pub fn reset(&mut self);
  }
  ```
- Turn frames into speaker segments with hysteresis (onset/offset thresholds, minimum segment length) so labels don't flicker.
- **Parity test:** a `cargo test` that streams the Phase 1 test WAV through the crate and asserts that logits match the Python references within tolerance and that segments match the reference RTTM.
- Add a tiny CLI binary (`diarize-cli file.wav`) for manual testing.
- Model files: download on first use into the app data dir (like Handy's model manager), with a checksum. Never commit weights.

🛑 **GATE 3:** Show test results and real-time factor from Rust on the Mac.

---

## 8. Phase 4: Meeting mode in the app

**Audio:** reuse Handy's capture pipeline, but in meeting mode keep a continuous stream instead of a push-to-talk buffer. Fan out the 16 kHz samples to:

- the **diarizer** (continuous, small chunks, on its own worker thread), and
- the **ASR worker**: transcribe each VAD-delimited utterance as soon as it ends (pseudo-streaming). Cap utterance length (e.g. 20–30 s) with a forced split so long monologues still appear. Use the user's selected Handy model. Parakeet V3 (multilingual, auto language detection) and Whisper should both be tested for Dutch.

**Timeline:** keep one monotonic sample clock as the single source of truth for both workers, so ASR word timestamps and diarizer frames live on the same timeline. Account for resampling and VAD offsets explicitly. This is where subtle bugs will hide.

**Alignment:** for each word (or segment, if an engine only gives segment timestamps), assign the speaker with the highest summed activity over the word's span. Words falling in diarizer gaps take the nearest speaker. Merge consecutive same-speaker words into turns. Speaker labels from the diarizer are arrival-ordered indices. Map them to stable display IDs (`Speaker 1..N`) plus user-assigned names.

**Late corrections:** diarizer output for recent frames can change until they're finalized. Render the last few seconds as "tentative" (e.g. lighter text) and lock turns once finalized.

**Wiring:**

- New CLI flag `--toggle-meeting`, a new shortcut binding, and a tray item "Start/Stop meeting".
- Meeting mode must **not** paste into the focused app.
- Settings page section: max speakers (1–8, default 6), latency profile, output folder, save audio (off by default), ASR model for meetings.

**Frontend:** a meeting window (React) subscribing to Tauri events: `meeting:turn_appended`, `meeting:turn_updated`, `meeting:speaker_renamed`, `meeting:stopped`. Click a speaker label to rename it. Show a recording timer and a speaker-count indicator.

**Output:** on stop, flush both workers, finalize, and write Markdown:

```markdown
# Meeting – 2026-10-26 14:00 (52 min)

Speakers: Jan, Speaker 2, Wouter

**Jan** [00:00:04]: …
**Wouter** [00:00:31]: …
```

Also write a sidecar JSON with word-level data for later use by an LLM summary step.

🛑 **GATE 4:** Demo checklist: a 10+ minute live test with 3 speakers on the Mac. Report latency from speech to on-screen text, CPU usage, memory over time, and observed speaker errors.

---

## 9. Phase 5 and later (don't start without my go-ahead)

- Speaker naming persisted per meeting; optional voice enrollment across meetings.
- "Summarise meeting" post-processing hook (local LLM or API, configurable).
- Overlap-aware transcription with `multitalker-parakeet-streaming` (one ASR instance per active speaker).
- Re-run offline diarization at the end with the 30.4 s profile for a higher-quality final transcript.
- Signed/notarised macOS build.

---

## 10. Working agreements

- **Plan before code.** At the start of each phase, write a short plan in `docs/meetscribe/PLAN.md` (append, dated). Then execute.
- **Small commits** with clear messages on the `meeting-mode` branch. Keep upstream files' diffs minimal.
- **Tests:** keep `cargo test` and the existing frontend tests green. Add tests for alignment logic (pure functions, easy to unit-test with synthetic timelines).
- **Be honest about numbers.** Record measured results, not expectations. If something didn't work, write that down.
- **Ask me** before: choosing a fallback route, adding a heavy dependency, changing upstream behaviour for normal dictation, pushing, or anything involving my real meeting recordings leaving the machine (they must not).
- If you get stuck for a long time on one issue, summarise what you tried and ask rather than thrashing.

## 11. References

- Handy repo: https://github.com/cjpais/Handy (README, BUILD.md, CLAUDE.md, AGENTS.md)
- Upstream stance on diarization: https://github.com/cjpais/Handy/discussions/599
- Nemotron 3 Diarization model card: https://huggingface.co/nvidia/Nemotron-3-Diarization
- NVIDIA blog post: https://huggingface.co/blog/nvidia/nemotron-diarization
- Multitalker Parakeet (later phase): https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
- Handy's single-speaker GGUF port (shows what was _not_ ported): https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf
- Streaming multi-speaker ASR trade-offs (cascaded vs multi-instance): https://arxiv.org/pdf/2609.10265
- Prior art using Nemotron 3 Diarization in a meeting app: https://github.com/estebanstifli/Meet2Notes
