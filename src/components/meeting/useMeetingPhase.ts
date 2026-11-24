// meetscribe: the meeting phase as the main window sees it, kept live from
// "meeting:status" events.
import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { toast } from "sonner";
import { commands, type MeetingPhase, type MeetingStatus } from "@/bindings";

export const useMeetingPhase = (): MeetingPhase => {
  const [phase, setPhase] = useState<MeetingPhase>("idle");
  useEffect(() => {
    commands.meetingStatus().then((s) => setPhase(s.phase));
    const unlisten = [
      listen<MeetingStatus>("meeting:status", (e) => setPhase(e.payload.phase)),
      // A start that fails before the meeting window opens has nowhere else
      // to show its reason.
      listen<string>("meeting:error", (e) => toast.error(e.payload)),
    ];
    return () => {
      unlisten.forEach((p) => p.then((f) => f()));
    };
  }, []);
  return phase;
};
