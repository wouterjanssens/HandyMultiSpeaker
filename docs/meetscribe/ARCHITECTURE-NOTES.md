# MeetScribe – architecture notes (Phase 0 recon)

Recon of upstream cjpais/Handy at `73c09585` ("feat: transcribe files into history", v0.9.8),
done 2026-10-26. Paths are relative to `src-tauri/src/` unless noted. Line numbers drift; treat
them as pointers, not contracts.

> **Upstream has moved since the handoff was written.** There is no `transcribe-rs`, `vad-rs` or
> direct `cpal` use in the app any more. Capture is the `handy-recorder` crate, and _every_ model
> (Whisper, Parakeet, Nemotron, VAD, …) runs through `transcribe-cpp` (GGUF/ggml, Metal on Mac),
> in a **separate worker process**.

---

## 1. Audio capture path

Crate: `handy-recorder 0.1` (wraps `handy-cpal`). Three threads (`handy-recorder/src/capture/engine.rs`):

| Thread                    | Does                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `handy-recorder-device`   | owns the cpal stream + watchdog; the callback writes device-rate frames into an SPSC ring (2 s)                                              |
| `handy-recorder-delivery` | drains ring → resamples (rubato `Fft`, 1024-frame blocks) → cuts exact chunks → `Sink::process_chunk`. **All app code incl. VAD runs here.** |
| `handy-recorder-notify`   | failure notifications                                                                                                                        |

App side (`managers/audio.rs`):

- `microphone_config()` → `RecorderConfig::speech()`: 16 kHz mono. `frames_per_chunk` = VAD frame
  (Silero 512 = 32 ms, Earshot 256 = 16 ms).
- The stream is opened by `start_microphone_stream()` and can stay open between recordings
  ("always-on"); when no sink is lent the delivery thread **discards** audio.
- `try_start_recording(binding_id, VadPolicy)` lends a fresh `CaptureSink`; `stop_recording()` →
  `CaptureSink::finish()` returns the `Vec<f32>`. Clips < 1 s are zero-padded to 1.25 s.

`CaptureSink::process_chunk` (`audio_toolkit/audio/sink.rs`):

- every chunk → level visualiser;
- `VadPolicy::Disabled` → every frame kept; otherwise `SmoothedVad` decides and **noise frames are
  dropped**. Pre-roll 450 ms is emitted as a burst at onset; onset 60 ms; hangover 450 ms
  (offline) / 1650 ms (streaming).
- Live tap: `on_audio: Arc<dyn Fn(&[f32])>` → `StreamRouter::feed` — **post-VAD frames only**.

Silero VAD runs **in-process** (CPU `transcribe_cpp::VadSession`), not in the worker.

### Timeline hazards (matter for Phase 4)

1. **No sample clock exists.** `AudioChunk` has no position; the recorder's frame counters are private.
   → Meeting mode must count samples itself in its sink.
2. **VAD drops samples.** Anything downstream of the existing sink lives on a compacted timeline.
   → Meeting mode taps **pre-VAD** and runs its own VAD for utterance segmentation.
3. **Ring overruns drop frames silently** (not zero-filled); the 16 kHz timeline compresses.
   Per-chunk overrun counts are not exposed. → Log `Stopped.dropped_frames` and accept/flag it; or
   upstream-patch handy-recorder to expose overruns.
4. **Resampler delay** (~171 samples ≈ 10.7 ms at 48→16 kHz) is not trimmed at start. Constant
   offset, harmless as long as both workers see the same stream.
5. A slow sink causes drops; a stuck sink trips `SinkStalled` after 10 s. → Meeting sink must only
   copy samples into channels and return.

## 2. Transcription

**Process isolation** (`engine_supervisor/`): the app re-launches its own exe with
`--transcribe-worker`; **one worker, one loaded model**, driven by a single owner thread.
Wire protocol: `[u32 json_len][json][u32 pcm_len][f32 LE pcm]`.
Requests: `Hello, Load, Run{RunOptions}+PCM, StreamBegin, Feed+PCM, Finalize, StreamReset`.
`Response::Transcript(transcribe_cpp::Transcript)` carries the **full** struct (segments, words,
tokens) — but the app only reads `.text` and `.language`.

Supervisor API (`engine_supervisor/supervisor.rs`): `transcribe(pcm, RunOptions) -> Transcript`,
`start_stream(...) -> StreamHandle`, `cancel()`.
`TranscriptionManager` (`managers/transcription.rs`): `transcribe_recording`, `start_stream`,
`transcribe_in_chunks`; `RunOptions.timestamps` defaults to `Auto` (richest available).

**Timestamps** (`transcribe_cpp::Transcript`, all `i64` **ms relative to the PCM of that run**):
`segments[] {t0_ms,t1_ms,text,speaker_id}`, `words[] {t0_ms,t1_ms,text}`, `tokens[]`,
`speaker_segments[]`.

| Granularity      | Families                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| token (+ words)  | Parakeet (incl. TDT v3, multitalker, Nemotron streaming), GigaAM, MedASR |
| word             | Granite                                                                  |
| **segment only** | **Whisper** (DTW word timing "not yet implemented"), MOSS                |
| none             | Canary, Cohere, Qwen3-ASR, Voxtral, SenseVoice, Moonshine, …             |

→ For per-word speaker alignment use **Parakeet V3** (multilingual, incl. Dutch). Whisper only
gives segment timing; we'd align per segment.

Long audio: `audio_toolkit/chunking.rs` cuts contiguous ~30 s chunks at pauses. **No timestamp
offsetting exists anywhere** — texts are just joined.

File transcription (latest upstream commit): `decode_audio_file` (sample-accurate, drops the
resampler delay) → WAV → `transcribe_in_chunks`. Also a headless CLI
`--transcribe-file <16k mono wav> [--model id] [--json]` (text only) — useful for test harnesses.

## 3. Recording state, shortcuts, tray, CLI

- `transcription_coordinator.rs`: pure state machine `Idle | Recording(binding) | Processing`,
  one coordinator thread, `Effect::Start/Stop` → `ACTION_MAP` (`actions.rs`). Only bindings
  `transcribe` / `transcribe_with_post_process` go through it.
- Other bindings: `shortcut/handler.rs` calls `action.start` on press, `action.stop` on release.
  → A meeting toggle should act in `start()` only (stop = no-op), **outside** the coordinator so it
  doesn't block dictation state.
- `TranscribeAction::stop` → transcribe → post-process → history → **paste** (spawned thread in
  `actions.rs`). Meeting mode simply never enters this path.
- New binding: insert default in `settings.rs::get_default_settings` (missing bindings are
  auto-merged into existing stores), add to `ACTION_MAP`, add i18n
  `settings.general.shortcut.bindings.<id>`, render `<ShortcutInput shortcutId=…/>`.
- **CLI**: `cli.rs` (clap derive) + `lib.rs` single-instance callback that matches **raw argv
  strings** (`--toggle-transcription`, …). `--toggle-meeting` = add the field + one `else if` arm.
  Note: toggle flags on the _first_ launch are currently ignored by upstream.
- Signals: SIGUSR2/SIGUSR1 → `signal_handle::send_transcription_input`.
- **Tray** (`tray.rs`): desired-state diffing (`MenuInputs` → `compute_desired` → `build_menu`),
  two layouts (busy/idle); click handlers in `lib.rs` `on_menu_event`. Strings are generated by
  `build.rs` from the `"tray"` object in each locale's `translation.json`.

## 4. Settings and frontend wiring

- `AppSettings` (`settings.rs`), `#[serde(default)]` per field + entry in `get_default_settings()`.
  Store: tauri-plugin-store `settings_store.json`, key `"settings"`. Schema version 2 + migrations.
- Expose: `#[tauri::command] #[specta::specta]` setter → register in `collect_commands![]`
  (`lib.rs`); typed events in `collect_events![]`. `src/bindings.ts` is regenerated **only when a
  debug build starts** (`bun run tauri dev`), not by `cargo build`.
- Frontend: `src/stores/settingsStore.ts` (`settingUpdaters` map), `hooks/useSettings.ts`; settings
  pages registered in `SECTIONS_CONFIG` (`src/components/Sidebar.tsx`). ESLint enforces i18n.
- **Windows**: main window created in code (`lib.rs`, label `main`); overlay is a second Vite entry
  (`src/overlay/index.html`, label `recording_overlay`). A meeting window = new Vite entry +
  `WebviewWindowBuilder` + **add `"meeting"` to `capabilities/default.json`** + branch the global
  `CloseRequested` handler (it hides every window and demotes the app on macOS).
- **Data dir** = Tauri `app_data_dir()` derived from the **bundle identifier** (or `<exe>/Data` in
  portable mode). ⚠ **Models are downloaded via `hf-hub` into the shared
  `~/.cache/huggingface/hub`**, so the fork shares model _files_ with an installed Handy even with a
  different identifier. Read sharing is fine; deleting a model in either app affects both.

## 5. Diarization already in the stack?

**Partly, and unused.** `transcribe-cpp 0.4.0` (now 0.4.1) has:

- a `DIARIZE` model role, `Model::diarize_session()` → `DiarizeSession::run(&pcm) -> Vec<SpeakerSegment>`;
  only family: **Sortformer v2.1, 4 speakers, 80 ms frames**; **one-shot over a whole buffer**
  (streaming AOSC/FIFO state is internal, no incremental feed);
- ASR-side `RunOptions.diarize` for MOSS, Granite 4.1 Plus, and the **multitalker Parakeet bundle**
  (runs embedded Sortformer over the clip, then one ASR pass per speaker) — batch only, not streaming.
- No speaker embeddings. The catalog filters out the diarize models; the worker protocol has no
  diarize request.
- Nemotron 3 Diarization support is an **open request** (handy-computer/transcribe.cpp#170).

## 6. Nemotron 3 Diarization – native runtime exists (outside Handy)

The model card's first "how to use" is **NVIDIA NeMo-Speech.cpp** (Apache-2.0, v0.2.0, 2026-10-02):
a ggml/llama.cpp-based C++ runtime with Metal support and a **stable C ABI** (`nemo_speech/diar.h`):

```c
nemo_speech_diar_create(cfg{model_path, gpu, preset, chunk/right_ctx/fifo/spkcache/update_period}, &model);
nemo_speech_diar_num_speakers(model);          // 8 for V3
nemo_speech_diar_seconds_per_frame(model);     // 0.01 for V3 high-res
nemo_speech_diar_stream_open(model, &stream);
nemo_speech_diar_stream_push_f32(stream, samples, n, 16000);
nemo_speech_diar_frame_count / frame_probs_start / frame_probs(stream, out, cap);  // raw per-frame probs
nemo_speech_diar_segments(stream, &seg_cfg{onset,offset,pad,min_gap,min_duration}, out, cap, &count);
nemo_speech_diar_stream_finish / close;
```

The HF repo ships `Nemotron-3-Diarization.q8_0.gguf` for it. This is essentially the
`StreamingDiarizer` the handoff planned to build via ONNX in Phases 2–3.

Model facts from `config.json`/`processor_config.json`: FastConformer encoder 31 layers × 512,
8× subsampling, 128 mel bins (n_fft 512, hop 160, win 400, preemph 0.97), streaming modes
`ultra_low_latency (3,1)`, `very_low_latency (6,2)`, `low_latency (9,4)`; default config
chunk 340 / right ctx 40 / FIFO 40 / spk-cache update 300. Licence **OpenMDW 1.1**, not gated.
Training data lists English, Mandarin, Indic languages and "21 languages" of one licensed set —
**Dutch not explicitly listed.**

Integration risk: NeMo-Speech.cpp vendors its own llama.cpp/ggml; transcribe-cpp vendors ggml too.
Linking both into one binary may clash (duplicate ggml symbols / Metal library). Mitigations: link
NeMo-Speech as a dylib with a two-level namespace, or run the diarizer in its own worker process
(same pattern as `engine_supervisor`).

## 7. Build status (Apple M5, macOS 27.0, CLT only)

- `cargo build` (debug) OK in 2m11s. `bun run tauri dev` runs; Metal backend initialises.
- Apple Intelligence is stubbed (needs full Xcode, not just Command Line Tools) — irrelevant to us.
- The dev build was run with `identifier = com.wouter.meetscribe` and `HANDY_DISABLE_UPDATER=1` so
  it never touched the installed Handy's `~/Library/Application Support/com.pais.handy`.
