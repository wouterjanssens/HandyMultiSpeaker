// meetscribe: start or stop a meeting from the main window.
import React from "react";
import { useTranslation } from "react-i18next";
import { commands } from "@/bindings";
import { Button } from "../ui/Button";
import { useMeetingPhase } from "./useMeetingPhase";

interface MeetingButtonProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

export const MeetingButton: React.FC<MeetingButtonProps> = ({
  size = "sm",
  className = "",
}) => {
  const { t } = useTranslation();
  const phase = useMeetingPhase();
  const running = phase === "running";
  const busy = phase === "starting" || phase === "stopping";

  const label =
    phase === "starting"
      ? t("meeting.starting")
      : phase === "stopping"
        ? t("meeting.stopping")
        : running
          ? t("meeting.stop")
          : t("meeting.start");

  return (
    <Button
      size={size}
      variant={running ? "danger" : "primary"}
      disabled={busy}
      onClick={() => commands.meetingToggle()}
      className={`inline-flex items-center gap-2 ${className}`}
    >
      <span
        aria-hidden="true"
        className={`size-2 rounded-full ${running ? "animate-pulse bg-background" : "bg-error"}`}
      />
      {label}
    </Button>
  );
};
