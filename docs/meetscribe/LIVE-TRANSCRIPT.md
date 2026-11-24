# MeetScribe – live transcript for AI tools (2026-11-19)

While a meeting runs, MeetScribe writes its transcript to `<meeting>.vtt` in the meetings folder,
so a person (`tail -f`) or an AI (Claude Code, Codex, any LLM CLI) can follow the meeting from a
file. `current.vtt` in the same folder points at the running meeting's file and exists only while
a meeting runs.

## Why WebVTT

A standard every LLM and many tools already understand: numbered cues, timestamps, speakers as
`<v Name>` voice tags, and `NOTE` blocks for anything else. No custom format to explain.

## The contract

- **Append-only while live.** A turn becomes a cue once it is locked (`TurnBuilder`): its
  speaker is certain and it can no longer grow or merge. Readers can remember the last cue number
  and read only what comes after it; nothing above ever changes.
- A turn locks when the next speaker starts, or once it is 20 s long (`TurnConfig::max_turn_ms`),
  so a monologue or two voices the live diarizer can't separate still reach the file. This applies
  to every meeting, local models included, and so also shows in the saved `.md`/`.json`: a long
  stretch by one speaker is saved as several consecutive turns of that speaker (each with its own
  timestamp) instead of one.
- **Speaker renames** are appended as `NOTE … is now called …`, not rewritten into earlier cues.
- **Stop** appends `NOTE meeting ended`; once the final transcript is known (OpenAI final speaker
  pass, names) the file is replaced once, atomically, by the final version ending in
  `NOTE final`, and `current.vtt` is removed. A `current.vtt` left by a crash is removed at the
  next start.

Code: `crates/meeting-core/src/vtt.rs` (format), `src-tauri/src/meeting/live.rs` (file, link),
`session.rs` (sync after every turn update, end at Stop, final version from `finish_after_stop` on the post-processing worker).

## Instructions for the AI

Settings → Meetings → "Use with your AI" copies a SKILL.md-style prompt
(`src-tauri/src/meeting/ai_instructions.md`) with the user's meetings folder filled in. It
explains the current session (`current.vtt`) versus the history, the file names, the live
contract and shell commands to follow a meeting incrementally (`awk` from cue N).
