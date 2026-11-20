// meetscribe: settings of the OpenAI entry in the model list (key, model, languages).
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useOpenAiStore } from "@/stores/openaiStore";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { SettingContainer } from "../../ui/SettingContainer";
import { Dropdown } from "../../ui/Dropdown";

const TRANSCRIBE_MODELS = ["gpt-transcribe", "gpt-4o-transcribe"];

/** Dropdown options, keeping a custom value already stored in the settings. */
export const withCurrent = (values: string[], current: string) =>
  (values.includes(current) ? values : [current, ...values]).map((v) => ({
    value: v,
    label: v,
  }));

export const OpenAiSettings: React.FC = () => {
  const { t } = useTranslation();
  const oa = useOpenAiStore((s) => s.settings);
  const key = useOpenAiStore((s) => s.key);
  const load = useOpenAiStore((s) => s.load);
  const update = useOpenAiStore((s) => s.update);
  const [languages, setLanguages] = useState("");

  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (oa) setLanguages(oa.languages.join(", "));
  }, [oa]);

  if (!oa) return null;

  return (
    <SettingsGroup
      title={t("openai.settings.title")}
      description={t("openai.settings.description")}
    >
      <SettingContainer
        title={t("openai.settings.key.label")}
        description={
          key?.found
            ? t("openai.settings.key.found", { source: key.source })
            : t("openai.settings.key.missing")
        }
        grouped={true}
      >
        <span className={key?.found ? "text-green-600" : "text-red-600"}>
          {key?.found ? "✓" : "✗"}
        </span>
      </SettingContainer>
      <SettingContainer
        title={t("openai.settings.transcribeModel.label")}
        description={t("openai.settings.transcribeModel.description")}
        grouped={true}
      >
        <Dropdown
          selectedValue={oa.transcribe_model}
          options={withCurrent(TRANSCRIBE_MODELS, oa.transcribe_model)}
          onSelect={(v) => update({ transcribe_model: v })}
        />
      </SettingContainer>
      <SettingContainer
        title={t("openai.settings.languages.label")}
        description={t("openai.settings.languages.description")}
        grouped={true}
      >
        <input
          value={languages}
          onChange={(e) => setLanguages(e.target.value)}
          onBlur={() =>
            update({
              languages: languages
                .split(/[\s,]+/)
                .map((l) => l.trim())
                .filter(Boolean),
            })
          }
          className="w-32 rounded-md border border-line bg-background px-2 py-1 text-sm"
        />
      </SettingContainer>
    </SettingsGroup>
  );
};
