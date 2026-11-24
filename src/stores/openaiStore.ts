// meetscribe: settings of the OpenAI model, shared by the Models page and the Meetings
// settings so they stay in sync. Whether OpenAI is used is the ordinary model selection.
import { create } from "zustand";
import {
  commands,
  type OpenAiKeyStatus,
  type OpenAiSettings,
} from "@/bindings";

interface OpenAiStore {
  settings: OpenAiSettings | null;
  key: OpenAiKeyStatus | null;
  load: () => Promise<void>;
  update: (patch: Partial<OpenAiSettings>) => Promise<OpenAiSettings | null>;
}

export const useOpenAiStore = create<OpenAiStore>()((set, get) => ({
  settings: null,
  key: null,
  load: async () => {
    const [settings, key] = await Promise.all([
      commands.getOpenaiSettings(),
      commands.openaiKeyStatus(),
    ]);
    set({ settings, key });
  },
  update: async (patch) => {
    const current = get().settings;
    if (!current) return null;
    const next = { ...current, ...patch };
    set({ settings: next });
    const r = await commands.setOpenaiSettings(next);
    const saved = r.status === "ok" ? r.data : current;
    set({ settings: saved });
    return saved;
  },
}));
