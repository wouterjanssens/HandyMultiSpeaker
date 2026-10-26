//! Display identities for diarizer speakers.
//!
//! The diarizer numbers speakers by first arrival (0, 1, …), which is already a stable order for
//! a single meeting. They are shown as "Speaker N" (N = index + 1) until the user names them;
//! a rename applies retroactively because turns store the index, not the label.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerMap {
    names: BTreeMap<u8, String>,
    /// Speakers that have been attributed at least one word, in first-seen order.
    seen: Vec<u8>,
}

impl SpeakerMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that `speaker` produced words; returns true the first time.
    pub fn observe(&mut self, speaker: u8) -> bool {
        if self.seen.contains(&speaker) {
            return false;
        }
        self.seen.push(speaker);
        true
    }

    pub fn seen(&self) -> &[u8] {
        &self.seen
    }

    /// Sets (or clears, with an empty/blank name) the user-facing name of `speaker`.
    pub fn rename(&mut self, speaker: u8, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            self.names.remove(&speaker);
        } else {
            self.names.insert(speaker, name.to_string());
        }
    }

    pub fn custom_name(&self, speaker: u8) -> Option<&str> {
        self.names.get(&speaker).map(String::as_str)
    }

    /// Label for display and export.
    pub fn label(&self, speaker: Option<u8>) -> String {
        match speaker {
            Some(s) => self
                .names
                .get(&s)
                .cloned()
                .unwrap_or_else(|| format!("Speaker {}", s as u32 + 1)),
            None => "Unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_labels_are_one_based() {
        let m = SpeakerMap::new();
        assert_eq!(m.label(Some(0)), "Speaker 1");
        assert_eq!(m.label(Some(7)), "Speaker 8");
        assert_eq!(m.label(None), "Unknown");
    }

    #[test]
    fn rename_and_clear() {
        let mut m = SpeakerMap::new();
        m.rename(1, "  Jan ");
        assert_eq!(m.label(Some(1)), "Jan");
        m.rename(1, "");
        assert_eq!(m.label(Some(1)), "Speaker 2");
    }

    #[test]
    fn observe_keeps_first_seen_order() {
        let mut m = SpeakerMap::new();
        assert!(m.observe(2));
        assert!(m.observe(0));
        assert!(!m.observe(2));
        assert_eq!(m.seen(), &[2, 0]);
    }
}
