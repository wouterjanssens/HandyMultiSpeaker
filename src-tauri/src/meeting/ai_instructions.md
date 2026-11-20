---
name: meetscribe-transcripts
description: Read MeetScribe meeting transcripts: the meeting running right now (a live file that keeps growing) and past meetings in the meetings folder. Use when the user asks about "the meeting", what was just said, who said what, decisions, action items, or wants a summary or answers during or after a meeting.
---

# MeetScribe meeting transcripts

MeetScribe records meetings and writes speaker-labelled transcripts to one folder:

**Meetings folder:** `{{DIR}}`

There are two kinds of meetings in it:

1. **The current meeting**: the one being recorded right now. Its transcript is still growing.
2. **Past meetings**: the history. Their files no longer change.

Read the files; never edit, move or delete them.

## Is a meeting running right now?

`{{DIR}}/current.vtt` exists **only while a meeting is running**. It points at the running
meeting's transcript file (a link, so reading it reads that file).

- `current.vtt` exists → a meeting is running; it is the current session.
- `current.vtt` does not exist → no meeting is running; everything in the folder is history.

## File names

Every meeting uses one name for all its files: `YYYY-MM-DD_HHMM_meeting` (start date and time,
local), with ` (2)`, ` (3)` … added when two meetings start in the same minute. When the user has
post-processing on, the files are renamed after the meeting ends to
`YYYY-MM-DD_HHMM_<Subject>` (e.g. `2026-11-19_1000_Release-planning`), so the subject is visible
in the folder. The date and time at the start of the name always stay.

| File           | What it is                                                                 |
| -------------- | -------------------------------------------------------------------------- |
| `….vtt`        | WebVTT transcript. Live while the meeting runs, final afterwards.          |
| `….md`         | Markdown transcript, written when the meeting ends.                        |
| `….cleaned.md` | The same transcript tidied by an AI (hesitations removed), when enabled.   |
| `….json`       | Word-level timings and speakers, written when the meeting ends (optional). |
| `…_notes.md`   | AI meeting notes (summary, decisions, action items), when enabled.         |
| `….wav`        | The recording, only if the user chose to keep audio.                       |

To find a past meeting, list the folder and pick by date and time (or subject) in the name; the
newest meeting sorts last. Older meetings may only have `.md`/`.json` (the `.vtt` exists since
November 2026).

## The live transcript (`.vtt`)

Standard WebVTT. Each finished line of speech is one numbered **cue**:

```
12
00:04:31.200 --> 00:04:36.900
<v Wouter>Let's take the intake numbers first.
```

- The number (`12`) is the cue number: 1, 2, 3 … in order, never reused.
- Times are from the start of the meeting (the header `NOTE` gives the clock time it started).
- `<v Name>` is the speaker. Unnamed speakers are `Speaker 1`, `Speaker 2` …; `Unknown` means no
  speaker could be attributed.

**While the meeting runs, the file is only ever appended to.** Nothing above the end changes.
More cues will keep arriving, typically a few seconds after the words are spoken (a line is only
written once its speaker is certain), so the newest few seconds of talk are not in the file yet.

- **Speaker renamed:** a line such as `NOTE Speaker 2 is now called Stefaan. This applies to
earlier cues too.` is appended. From then on read `Speaker 2` in earlier cues as `Stefaan`.
- **Meeting ended:** the file ends with `NOTE meeting ended`. Shortly afterwards it is replaced
  once by the **final** transcript, with the final speaker names everywhere (and, with OpenAI,
  corrected speaker labels), ending with `NOTE final`. Cue numbers in the final file can differ
  from the live ones, so read the final file from the top.

## Following a meeting that is running

1. Note which file `current.vtt` points to (e.g. `2026-11-19_1000_meeting.vtt`): `current.vtt`
   disappears when the meeting ends, and that file is where the transcript stays (only its name
   may change once more, to the meeting's subject). Read the whole
   file once and remember the **last cue number** you saw.
2. Later, read only what came after that number (commands below) instead of the whole file
   again. Keep going while the last line is not `NOTE meeting ended` and `current.vtt` still
   exists. If `current.vtt` now points to a different file, a new meeting has started.
3. When it ends, wait a minute (the final speaker pass, notes and a possible rename to the
   subject run first), then find the meeting in the folder by its date and time and read its
   `.vtt` (now final), `.md` and, if there is one, `_notes.md`.

Check every 15–30 seconds while following; there is no need to check more often.

Commands (macOS/Linux shell):

```sh
# Is a meeting running, and which file is it?
ls -l "{{DIR}}/current.vtt"

# Everything so far
cat "{{DIR}}/current.vtt"

# Only what came after cue N (here 42), plus any NOTEs after it
awk -v n=42 'BEGIN{RS="";ORS="\n\n"} p{print} {split($0,l,"\n")} l[1]==n{p=1}' "{{DIR}}/current.vtt"

# Last few lines: has the meeting ended?
tail -n 3 "{{DIR}}/current.vtt"

# Watch it grow (for a person in a terminal)
tail -f "{{DIR}}/current.vtt"
```

Windows PowerShell: `Get-Content "{{DIR}}\current.vtt"` reads it, `Get-Content -Wait` follows it.

If you cannot run commands, ask the user to paste the file, or the part after the last cue you
have seen.

## Answering about a meeting

- Quote who said what using the speaker names, with the time (`04:31`) when it helps.
- Say when something is uncertain because the meeting is still running, e.g. "as of 10:42, cue
  57".
- For past meetings, prefer `_notes.md` for a quick overview and the `.vtt` or `.md` for exact
  wording.
