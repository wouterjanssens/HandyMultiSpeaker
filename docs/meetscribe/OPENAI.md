# MeetScribe – OpenAI engine (2026-11-17)

An optional cloud route for machines without a fast GPU (first target: an i7-8550U laptop with
Intel UHD 620, Linux). **OpenAI is a model in the ordinary model list** (id `openai`, source
`Cloud`): nothing to download, nothing loaded into memory, never auto-selected. Pick it in the
footer model picker, the tray or on the Models page and dictation (and History/file
transcriptions) go to OpenAI. Meetings follow the dictation model ("Same as dictation") unless
**Meetings → Meeting transcription model** picks one explicitly — OpenAI is in that list too.
Audio only leaves the machine while OpenAI is the model in use.

Settings (stored in `openai_settings.json`; the first version kept them under the `openai` key of
`meeting_settings.json` with a `dictation` flag, migrated once at startup into the model
selection):

- **Models page**, shown while OpenAI is the model in use: API key status, transcription model,
  spoken languages.
- **Meetings page**, shown while meetings run on OpenAI: final speaker pass, AI notes, notes model
  (plus the model settings when dictation uses a local model).

## What runs where

| Step                         | Model / API                                        | When                       |
| ---------------------------- | -------------------------------------------------- | -------------------------- |
| Utterance segmentation (VAD) | local Silero, unchanged                            | live                       |
| Text                         | `gpt-transcribe` (`languages[]`, `keywords[]`)     | live, per utterance        |
| Live speakers                | `gpt-4o-transcribe-diarize` + ≤ 4 enrolled voices  | live, per utterance        |
| Final speakers               | `gpt-4o-transcribe-diarize` on the whole recording | after Stop (10 min chunks) |
| Notes                        | Responses API, `gpt-5.5`                           | after Stop → `…_notes.md`  |
| Dictation                    | `gpt-transcribe`                                   | on release of the shortcut |

Final speakers and notes only run for meetings on the OpenAI model, and never inside Stop: Stop
saves the transcript (live speaker labels) at once and hands the rest to the meeting's
post-processing worker, which runs the final speaker pass, rewrites the `.md`, `.json` and History
entry, writes the final `.vtt`, then the notes, and only then the LLM clean-up and rename. Without
"save audio" the final pass records to `meetscribe-*.wav` in the temp folder; it is deleted after
the pass, and any left by a crash are removed at the next start.

Code: `src-tauri/src/openai/` (client, settings, speaker tracker, notes),
`src-tauri/src/meeting/cloud.rs` (live pipeline), `src-tauri/src/meeting/refine.rs` (final pass).
Settings are stored in `openai_settings.json` (key `openai`).

API key lookup order: `OPENAI_API_KEY`, the OS keyring (`secret-tool lookup service openai account
api-key` on Linux; keychain item `openai`/`api-key` on macOS), then the Post Processing → OpenAI key.

## Why this split (measured 2026-11-17)

- `gpt-transcribe` gives the best text (keywords fixed "Go4TMS", "Productive") but returns **no
  timestamps** (`verbose_json` is rejected) and no speakers. ~0.5–1 s per utterance.
- `gpt-4o-transcribe-diarize` gives speakers with times but weaker text ("GO for t m s", "B say
  team") and hallucinates on very short clips without a language hint, so only its labels are used.
- Per-utterance diarization with known-speaker references keeps labels consistent across requests,
  but with only a few seconds of context it merged two similar male voices. The same model on the
  whole recording separated all three speakers correctly → the final pass.

Simulated 83 s, 3-speaker NL/EN meeting (TTS voices onyx/nova/fable), real-time through the app
(`MEETSCRIBE_SIMULATE_WAV`):

| Metric                      | Result                                    |
| --------------------------- | ----------------------------------------- |
| Speech end → text on screen | median 1.0 s, p95 1.3 s                   |
| Live speaker labels         | 3 voices enrolled; one merge before fixes |
| Final pass                  | 40 s, 10/10 turns on the right speaker    |
| Notes (`gpt-5.5`)           | 9 s, Dutch, correct decisions and owners  |

Cost (Nov 2026 list prices): text $0.0045/min + diarization $0.006/min live + $0.006/min final,
≈ $1/hour of meeting plus a few cents for notes.

## Limits

- `gpt-4o-transcribe-diarize` is **deprecated; OpenAI shuts it down on 2027-02-26**. No
  non-deprecated diarization route existed when this was written; revisit before then.
- Live: at most four voices can be enrolled (API limit); a fifth speaker shows as unknown until
  the final pass.
- Final pass on a long meeting: chunks after the first are matched to the first chunk's four main
  voices; a speaker who only appears later is labelled per chunk.
- Audio is uploaded as 16 kHz WAV (≤ 10 min per request, under the 25 MB limit).
