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
  type SpeakerDto,
  type TurnDto,
} from "@/bindings";
import { Button } from "@/components/ui/Button";

/** Emitted on `meeting:stopped` (not part of the generated bindings). */
interface MeetingOutput {
  markdown_path: string | null;
  json_path: string | null;
  audio_path: string | null;
  error: string | null;
}

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

/** Same layout as the saved Markdown transcript, with current speaker names. */
const transcriptText = (turns: TurnDto[]) =>
  turns
    .filter((t) => t.text.trim().length > 0)
    .map((t) => `**${t.label}** [${hms(t.start_ms)}]: ${t.text}`)
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
  const scroller = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // initial state for a window opened mid-meeting, then live events
  useEffect(() => {
    commands.meetingStatus().then(setStatus);
    commands.meetingTurns().then(setTurns);
    const unlisten = [
      listen<MeetingStatus>("meeting:status", (e) => {
        setStatus(e.payload);
        if (e.payload.phase === "starting") {
          setTurns([]);
          setOutput(null);
          setError(null);
        }
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

  const copyTranscript = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(transcriptText(turns));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      setError(String(e));
    }
  }, [turns]);

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
            <p className="leading-relaxed">{turn.text}</p>
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
        <span className="ml-auto truncate" title={status.diarizer ?? ""}>
          {status.asr_model ?? ""}
        </span>
      </footer>
    </div>
  );
};

export default MeetingWindow;
