import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  commands,
  type LatencyProfile,
  type MeetingSettings as MeetingSettingsT,
  type ModelInfo,
} from "@/bindings";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { SettingContainer } from "../../ui/SettingContainer";
import { Dropdown } from "../../ui/Dropdown";
import { Slider } from "../../ui/Slider";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { Button } from "../../ui/Button";
import { ShortcutInput } from "../ShortcutInput";

const LATENCIES: LatencyProfile[] = ["ultra_low", "very_low", "low", "high"];

/** Meeting-mode settings; stored by the backend in their own file. */
export const MeetingSettings: React.FC = () => {
  const { t } = useTranslation();
  const [settings, setSettings] = useState<MeetingSettingsT | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [outputDir, setOutputDir] = useState("");

  useEffect(() => {
    commands.getMeetingSettings().then((s) => {
      setSettings(s);
      setOutputDir(s.output_dir ?? "");
    });
    commands.getAvailableModels().then((r) => {
      if (r.status === "ok") setModels(r.data.filter((m) => m.is_downloaded));
    });
  }, []);

  const update = useCallback(
    async (patch: Partial<MeetingSettingsT>) => {
      if (!settings) return;
      const next = { ...settings, ...patch };
      setSettings(next);
      const r = await commands.setMeetingSettings(next);
      if (r.status === "ok") setSettings(r.data);
      else setSettings(settings);
    },
    [settings],
  );

  if (!settings) return null;

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.meeting.title")}>
        <ShortcutInput shortcutId="toggle_meeting" grouped={true} />
        <SettingContainer
          title={t("settings.meeting.open.label")}
          description={t("settings.meeting.open.description")}
          grouped={true}
        >
          <Button
            size="sm"
            variant="secondary"
            onClick={() => commands.meetingOpenWindow()}
          >
            {t("settings.meeting.open.button")}
          </Button>
        </SettingContainer>
        <Slider
          value={settings.max_speakers}
          min={1}
          max={8}
          step={1}
          onChange={(v) => update({ max_speakers: v })}
          label={t("settings.meeting.maxSpeakers.label")}
          description={t("settings.meeting.maxSpeakers.description")}
          grouped={true}
          formatValue={(v) => v.toString()}
        />
        <SettingContainer
          title={t("settings.meeting.latency.label")}
          description={t("settings.meeting.latency.description")}
          grouped={true}
        >
          <Dropdown
            selectedValue={settings.latency_profile}
            options={LATENCIES.map((l) => ({
              value: l,
              label: t(`settings.meeting.latency.${l}`),
            }))}
            onSelect={(v) => update({ latency_profile: v as LatencyProfile })}
          />
        </SettingContainer>
        <SettingContainer
          title={t("settings.meeting.asrModel.label")}
          description={t("settings.meeting.asrModel.description")}
          grouped={true}
        >
          <Dropdown
            selectedValue={settings.asr_model ?? ""}
            options={[
              { value: "", label: t("settings.meeting.asrModel.same") },
              ...models.map((m) => ({ value: m.id, label: m.name })),
            ]}
            onSelect={(v) => update({ asr_model: v || null })}
          />
        </SettingContainer>
      </SettingsGroup>
      <SettingsGroup title={t("meeting.savedTo")}>
        <SettingContainer
          title={t("settings.meeting.outputDir.label")}
          description={t("settings.meeting.outputDir.description")}
          grouped={true}
          layout="stacked"
        >
          <div className="flex gap-2">
            <input
              value={outputDir}
              placeholder={t("settings.meeting.outputDir.placeholder")}
              onChange={(e) => setOutputDir(e.target.value)}
              onBlur={() => update({ output_dir: outputDir.trim() || null })}
              className="flex-1 rounded-md border border-line bg-background px-2 py-1 text-sm"
            />
            <Button
              size="sm"
              variant="ghost"
              onClick={() => commands.meetingRevealOutput(null)}
            >
              {t("meeting.openFolder")}
            </Button>
          </div>
        </SettingContainer>
        <ToggleSwitch
          checked={settings.write_json}
          onChange={(v) => update({ write_json: v })}
          label={t("settings.meeting.writeJson.label")}
          description={t("settings.meeting.writeJson.description")}
          grouped={true}
        />
        <ToggleSwitch
          checked={settings.save_audio}
          onChange={(v) => update({ save_audio: v })}
          label={t("settings.meeting.saveAudio.label")}
          description={t("settings.meeting.saveAudio.description")}
          grouped={true}
        />
      </SettingsGroup>
    </div>
  );
};
