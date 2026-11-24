import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import {
  commands,
  type MeetingStatus,
  type PostProcessMode,
  type PostProcessOptions,
  type SpeakerDto,
  type TurnDto,
} from "@/bindings";
import { Button } from "@/components/ui/Button";
import { Dropdown } from "@/components/ui/Dropdown";

/** Spoken-language hints; names in their own language, "auto" translated. */
const LANGUAGES: { value: string; label: string | null }[] = [
  { value: "auto", label: null },
  { value: "en", label: "English" },
  { value: "nl", label: "Nederlands" },
  { value: "da", label: "Dansk" },
  { value: "fr", label: "Français" },
];

/** Emitted on `meeting:stopped` (not part of the generated bindings). */
interface MeetingOutput {
  markdown_path: string | null;
  json_path: string | null;
  audio_path: string | null;
  notes_path: string | null;
  error: string | null;
}

/** Emitted on `meeting:turn_cleaned`; `meeting` is that meeting's start time. */
interface TurnCleaned {
  meeting: number;
  id: number;
  text: string;
}

/** Emitted on `meeting:post_process` while a stopped meeting is cleaned up. */
interface PostProcessProgress {
  meeting: number;
  done: number;
  total: number;
  finished: boolean;
  path: string | null;
  /** The meeting's files under their new, titled names. */
  renamed: MeetingOutput | null;
  error: string | null;
}

const POST_PROCESS_MODES: PostProcessMode[] = ["off", "at_end", "continuous"];

interface SpeakerRenamed {
  speaker: number;
  label: string;
}

const IDLE: MeetingStatus = {
  phase: "idle",
  started_epoch_ms: null,
  audio_ms: 0,
  speakers: [],
  asr_model: null,
  diarizer: null,
  diarizer_latency_ms: 0,
  stats: null,
  language_ignored: false,
};

/** Distinct, theme-neutral hues for up to 8 arrival-ordered speakers. */
const SPEAKER_HUES = [210, 25, 150, 290, 0, 55, 180, 330];
const speakerColor = (speaker: number | null) =>
  speaker === null
    ? "var(--color-ink-3)"
    : `hsl(${SPEAKER_HUES[speaker % SPEAKER_HUES.length]} 65% 48%)`;

const hms = (ms: number) => {
  const s = Math.max(0, Math.floor(ms / 1000));
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}`;
};

const shownText = (turn: TurnDto, original: boolean) =>
  original ? turn.text : (turn.cleaned ?? turn.text);

/** Same layout as the saved Markdown transcript, with current speaker names. */
const transcriptText = (turns: TurnDto[], original: boolean) =>
  turns
    .filter((t) => t.text.trim().length > 0)
    .map(
      (t) => `**${t.label}** [${hms(t.start_ms)}]: ${shownText(t, original)}`,
    )
    .join("\n\n");

const upsert = (turns: TurnDto[], turn: TurnDto) => {
  const i = turns.findIndex((t) => t.id === turn.id);
  if (i === -1) return [...turns, turn].sort((a, b) => a.id - b.id);
  const next = turns.slice();
  next[i] = turn;
  return next;
};

const SpeakerChip: React.FC<{
  speaker: SpeakerDto;
  onRename: (id: number, name: string) => void;
}> = ({ speaker, onRename }) => {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(speaker.custom_name ?? "");

  if (editing) {
    return (
      <form
        className="flex items-center gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          onRename(speaker.id, value);
          setEditing(false);
        }}
      >
        <input
          autoFocus
          value={value}
          placeholder={t("meeting.renamePlaceholder")}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && setEditing(false)}
          className="w-32 rounded-md border border-line bg-background px-2 py-0.5 text-sm"
        />
        <Button size="sm" variant="primary-soft" type="submit">
          {t("meeting.save")}
        </Button>
      </form>
    );
  }
  return (
    <button
      type="button"
      title={t("meeting.renameHint")}
      onClick={() => {
        setValue(speaker.custom_name ?? "");
        setEditing(true);
      }}
      className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-0.5 text-sm hover:border-line-strong"
    >
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ background: speakerColor(speaker.id) }}
      />
      {speaker.label}
    </button>
  );
};

const MeetingWindow: React.FC = () => {
  const { t } = useTranslation();
  const [status, setStatus] = useState<MeetingStatus>(IDLE);
  const [turns, setTurns] = useState<TurnDto[]>([]);
  const [output, setOutput] = useState<MeetingOutput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [copied, setCopied] = useState(false);
  const [language, setLanguage] = useState("auto");
  const [mode, setMode] = useState<PostProcessMode>("off");
  const [promptId, setPromptId] = useState<string | null>(null);
  const [options, setOptions] = useState<PostProcessOptions | null>(null);
  const [showEnableHelp, setShowEnableHelp] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const [cleanup, setCleanup] = useState<PostProcessProgress | null>(null);
  // start time of the meeting on screen; late results of an earlier one are dropped
  const meetingId = useRef<number | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // initial state for a window opened mid-meeting, then live events
  useEffect(() => {
    commands.meetingStatus().then(setStatus);
    commands.meetingStatus().then((s) => {
      if (s.started_epoch_ms !== null) meetingId.current = s.started_epoch_ms;
    });
    commands.getMeetingSettings().then((s) => {
      setLanguage(s.language);
      setMode(s.post_process_mode);
      setPromptId(s.post_process_prompt_id);
    });
    commands.meetingPostProcessOptions().then(setOptions);
    commands.meetingTurns().then(setTurns);
    const unlisten = [
      listen<MeetingStatus>("meeting:status", (e) => {
        setStatus(e.payload);
        if (e.payload.started_epoch_ms !== null)
          meetingId.current = e.payload.started_epoch_ms;
        if (e.payload.phase === "starting") {
          meetingId.current = null;
          setTurns([]);
          setOutput(null);
          setError(null);
          setCleanup(null);
        }
      }),
      listen<TurnCleaned>("meeting:turn_cleaned", (e) => {
        const { meeting, id, text } = e.payload;
        if (meeting !== meetingId.current) return;
        setTurns((ts) =>
          ts.map((t) => (t.id === id ? { ...t, cleaned: text } : t)),
        );
      }),
      listen<PostProcessProgress>("meeting:post_process", (e) => {
        if (e.payload.meeting !== meetingId.current) return;
        setCleanup(e.payload);
        if (e.payload.renamed) setOutput(e.payload.renamed);
        if (e.payload.error) setError(e.payload.error);
      }),
      listen<TurnDto>("meeting:turn_appended", (e) =>
        setTurns((ts) => upsert(ts, e.payload)),
      ),
      listen<TurnDto>("meeting:turn_updated", (e) =>
        setTurns((ts) => upsert(ts, e.payload)),
      ),
      listen<number>("meeting:turn_removed", (e) =>
        setTurns((ts) => ts.filter((t) => t.id !== e.payload)),
      ),
      listen<SpeakerRenamed>("meeting:speaker_renamed", (e) => {
        const { speaker, label } = e.payload;
        setTurns((ts) =>
          ts.map((t) => (t.speaker === speaker ? { ...t, label } : t)),
        );
        setStatus((s) => ({
          ...s,
          speakers: s.speakers.map((sp) =>
            sp.id === speaker ? { ...sp, label, custom_name: label } : sp,
          ),
        }));
      }),
      listen<MeetingOutput>("meeting:stopped", (e) => {
        setOutput(e.payload);
        if (e.payload.error) setError(e.payload.error);
      }),
      listen<string>("meeting:error", (e) => setError(e.payload)),
    ];
    return () => {
      unlisten.forEach((p) => p.then((f) => f()));
    };
  }, []);

  useEffect(() => {
    if (status.phase !== "running") return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [status.phase]);

  useEffect(() => {
    const el = scroller.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const onScroll = useCallback(() => {
    const el = scroller.current;
    if (el)
      stickToBottom.current =
        el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }, []);

  const rename = useCallback(async (id: number, name: string) => {
    const r = await commands.meetingRenameSpeaker(id, name);
    if (r.status === "error") setError(r.error);
  }, []);

  const changeLanguage = useCallback(
    async (value: string) => {
      const previous = language;
      setLanguage(value);
      const r = await commands.meetingSetLanguage(value);
      if (r.status === "error") {
        setLanguage(previous);
        setError(r.error);
      }
    },
    [language],
  );

  // Saves the choice; turning it on needs the experimental post-processing feature, which may
  // have been switched on or off in the main window since this one opened.
  const changePostProcess = useCallback(
    async (nextMode: PostProcessMode, nextPrompt: string | null) => {
      const latest = await commands.meetingPostProcessOptions();
      setOptions(latest);
      if (nextMode !== "off" && !latest.enabled) {
        setShowEnableHelp(true);
        return;
      }
      setShowEnableHelp(false);
      const r = await commands.meetingSetPostProcess(nextMode, nextPrompt);
      if (r.status === "error") {
        setError(r.error);
        return;
      }
      setMode(r.data.post_process_mode);
      setPromptId(r.data.post_process_prompt_id);
    },
    [],
  );

  const copyTranscript = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(transcriptText(turns, showOriginal));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      setError(String(e));
    }
  }, [turns, showOriginal]);

  const hasCleaned = useMemo(
    () => turns.some((turn) => turn.cleaned !== null),
    [turns],
  );
  const prompts = options?.prompts ?? [];
  const selectedPrompt =
    prompts.find((p) => p.id === promptId)?.id ??
    prompts.find((p) => p.id === options?.default_prompt_id)?.id ??
    null;

  const running = status.phase === "running";
  const busy = status.phase === "starting" || status.phase === "stopping";
  const elapsed =
    running && status.started_epoch_ms !== null
      ? now - status.started_epoch_ms
      : status.audio_ms;

  // speakers from the status plus any seen in turns (status refreshes ~1/s)
  const speakers = useMemo(() => {
    const byId = new Map(status.speakers.map((s) => [s.id, s]));
    for (const turn of turns) {
      if (turn.speaker !== null && !byId.has(turn.speaker)) {
        byId.set(turn.speaker, {
          id: turn.speaker,
          label: turn.label,
          custom_name: null,
        });
      }
    }
    return [...byId.values()].sort((a, b) => a.id - b.id);
  }, [status.speakers, turns]);

  return (
    <div className="flex h-screen flex-col bg-background text-text">
      <header className="flex items-center gap-3 border-b border-line px-4 py-3">
        <span
          className={`h-3 w-3 rounded-full ${running ? "animate-pulse" : ""}`}
          style={{
            background: running
              ? "var(--color-error)"
              : busy
                ? "var(--color-warning)"
                : "var(--color-line-strong)",
          }}
        />
        <h1 className="text-lg font-semibold">{t("meeting.title")}</h1>
        <span className="font-mono text-sm tabular-nums text-ink-2">
          {hms(elapsed)}
        </span>
        <span className="text-sm text-ink-3">
          {status.phase === "starting" && t("meeting.starting")}
          {status.phase === "stopping" && t("meeting.stopping")}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="shrink-0" title={t("meeting.languageHint")}>
            <Dropdown
              selectedValue={language}
              options={LANGUAGES.map((l) => ({
                value: l.value,
                label: l.label ?? t("meeting.languageAuto"),
              }))}
              onSelect={changeLanguage}
            />
          </div>
          <Button
            variant="secondary"
            disabled={turns.length === 0}
            onClick={copyTranscript}
          >
            {copied ? t("meeting.copied") : t("meeting.copy")}
          </Button>
          <Button
            variant={running ? "danger" : "primary"}
            disabled={busy}
            onClick={() => commands.meetingToggle()}
          >
            {running ? t("meeting.stop") : t("meeting.start")}
          </Button>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2">
        <span
          className="text-xs uppercase tracking-wide text-ink-3"
          title={t("meeting.postProcess.hint")}
        >
          {t("meeting.postProcess.label")}
        </span>
        <Dropdown
          selectedValue={mode}
          options={POST_PROCESS_MODES.map((value) => ({
            value,
            label: t(`meeting.postProcess.mode.${value}`),
          }))}
          onSelect={(value) =>
            changePostProcess(value as PostProcessMode, promptId)
          }
        />
        <Dropdown
          selectedValue={selectedPrompt}
          options={prompts.map((p) => ({ value: p.id, label: p.name }))}
          placeholder={t("meeting.postProcess.noPrompts")}
          disabled={prompts.length === 0}
          onOpen={() => commands.meetingPostProcessOptions().then(setOptions)}
          onSelect={(value) => changePostProcess(mode, value)}
        />
        {hasCleaned && (
          <label className="ml-auto flex items-center gap-1.5 text-sm text-ink-2">
            <input
              type="checkbox"
              checked={showOriginal}
              onChange={(e) => setShowOriginal(e.target.checked)}
            />
            {t("meeting.postProcess.showOriginal")}
          </label>
        )}
      </div>

      {showEnableHelp && (
        <div className="mx-4 mt-3 flex items-start gap-3 rounded-lg border border-line bg-warning-wash px-3 py-2 text-sm">
          <div className="flex-1">
            <div className="font-medium">
              {t("meeting.postProcess.disabledTitle")}
            </div>
            <div className="text-ink-2">
              {t("meeting.postProcess.disabledHelp")}
            </div>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setShowEnableHelp(false)}
          >
            {t("meeting.dismiss")}
          </Button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2">
        <span className="text-xs uppercase tracking-wide text-ink-3">
          {t("meeting.speakers")}
        </span>
        {speakers.length === 0 ? (
          <span className="text-sm text-ink-3">{t("meeting.noSpeakers")}</span>
        ) : (
          speakers.map((s) => (
            <SpeakerChip key={s.id} speaker={s} onRename={rename} />
          ))
        )}
      </div>

      {status.language_ignored && (
        <div className="mx-4 mt-3 rounded-lg border border-line bg-warning-wash px-3 py-2 text-sm text-ink-2">
          {t("meeting.languageIgnored")}
        </div>
      )}

      {error && (
        <div className="mx-4 mt-3 flex items-start gap-3 rounded-lg border border-line bg-error-wash px-3 py-2 text-sm">
          <div className="flex-1">
            <div className="font-medium">{t("meeting.errorTitle")}</div>
            <div className="text-ink-2">{error}</div>
          </div>
          <Button size="sm" variant="ghost" onClick={() => setError(null)}>
            {t("meeting.dismiss")}
          </Button>
        </div>
      )}

      <div
        ref={scroller}
        onScroll={onScroll}
        className="flex-1 space-y-3 overflow-y-auto px-4 py-4"
      >
        {turns.length === 0 && (
          <p className="mt-8 text-center text-sm text-ink-3">
            {running || busy ? t("meeting.waiting") : t("meeting.idleHint")}
          </p>
        )}
        {turns.map((turn) => (
          <div
            key={turn.id}
            className="transition-opacity"
            style={{ opacity: turn.locked ? 1 : 0.6 }}
          >
            <div className="flex items-baseline gap-2">
              <span
                className="text-sm font-semibold"
                style={{ color: speakerColor(turn.speaker) }}
              >
                {turn.label}
              </span>
              <span className="font-mono text-xs tabular-nums text-ink-3">
                {hms(turn.start_ms)}
              </span>
            </div>
            <p className="leading-relaxed">{shownText(turn, showOriginal)}</p>
          </div>
        ))}
      </div>

      <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line px-4 py-2 text-xs text-ink-3">
        {status.stats && running && (
          <>
            <span>
              {t("meeting.asrDelay", { ms: status.stats.last_asr_delay_ms })}
            </span>
            <span>
              {t("meeting.diarizerLag", { ms: status.stats.diarizer_lag_ms })}
            </span>
            <span>
              {t("meeting.utterances", { count: status.stats.utterances })}
            </span>
          </>
        )}
        {output?.markdown_path && (
          <span className="flex items-center gap-2">
            {t("meeting.savedTo")}:{" "}
            <span className="font-mono text-ink-2">
              {output.markdown_path.split(/[\\/]/).pop()}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => commands.meetingRevealOutput(output.markdown_path)}
            >
              {t("meeting.reveal")}
            </Button>
          </span>
        )}
        {cleanup && !cleanup.finished && (
          <span>
            {t("meeting.postProcess.progress", {
              done: cleanup.done,
              total: cleanup.total,
            })}
          </span>
        )}
        {cleanup?.path && (
          <span className="flex items-center gap-2">
            {t("meeting.postProcess.savedTo")}:{" "}
            <span className="font-mono text-ink-2">
              {cleanup.path.split(/[\\/]/).pop()}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => commands.meetingRevealOutput(cleanup.path)}
            >
              {t("meeting.reveal")}
            </Button>
          </span>
        )}
        {output?.notes_path && (
          <span className="flex items-center gap-2">
            {t("meeting.notesSaved")}:{" "}
            <span className="font-mono text-ink-2">
              {output.notes_path.split(/[\\/]/).pop()}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => commands.meetingRevealOutput(output.notes_path)}
            >
              {t("meeting.reveal")}
            </Button>
          </span>
        )}
        <span className="ml-auto truncate" title={status.diarizer ?? ""}>
          {status.asr_model ?? ""}
        </span>
      </footer>
    </div>
  );
};

export default MeetingWindow;
