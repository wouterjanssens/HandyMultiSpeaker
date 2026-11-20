// meetscribe: instructions (a SKILL.md) that let any AI read the meetings folder, live and past.
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands } from "@/bindings";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { Button } from "../../ui/Button";

export const AiInstructions: React.FC<{ outputDir: string | null }> = ({
  outputDir,
}) => {
  const { t } = useTranslation();
  const [text, setText] = useState("");
  const [shown, setShown] = useState(false);
  const [copied, setCopied] = useState(false);

  // the folder is part of the text, so fetch again when it changes
  useEffect(() => {
    commands.meetingAiInstructions().then(setText);
  }, [outputDir]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Copy failed:", e);
    }
  };

  return (
    <SettingsGroup
      title={t("settings.meeting.ai.title")}
      description={t("settings.meeting.ai.description")}
    >
      <div className="flex flex-wrap items-center gap-2 px-4 py-3">
        <Button size="md" variant="primary" onClick={copy} disabled={!text}>
          {copied
            ? t("settings.meeting.ai.copied")
            : t("settings.meeting.ai.copy")}
        </Button>
        <Button
          size="md"
          variant="secondary"
          onClick={() => setShown((s) => !s)}
        >
          {shown
            ? t("settings.meeting.ai.hide")
            : t("settings.meeting.ai.show")}
        </Button>
        <span className="text-xs text-ink-3">
          {t("settings.meeting.ai.hint")}
        </span>
      </div>
      {shown && (
        <pre className="mx-4 mb-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-md border border-line bg-wash p-3 text-xs select-text">
          {text}
        </pre>
      )}
    </SettingsGroup>
  );
};
