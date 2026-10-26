//! Display identities for diarizer speakers.
//!
//! The diarizer numbers speakers by first *acoustic* arrival (0, 1, …). The first voice it hears
//! may never produce transcribed words (a cough, a short "yes" the ASR drops), so display numbers
//! follow first appearance in the transcript instead: whoever is attributed words first is
//! "Speaker 1". Callers [`SpeakerMap::observe`] each speaker as its words reach the transcript.
//! A rename applies retroactively because turns store the diarizer index, not the label.

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
            Some(s) => self.names.get(&s).cloned().unwrap_or_else(|| {
                // unobserved speakers (shouldn't be displayed) fall back to their raw index
                let n = self
                    .seen
                    .iter()
                    .position(|&x| x == s)
                    .map_or(s as usize, |pos| pos)
                    + 1;
                format!("Speaker {n}")
            }),
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
    fn display_numbers_follow_first_appearance() {
        let mut m = SpeakerMap::new();
        m.observe(2);
        m.observe(0);
        assert_eq!(m.label(Some(2)), "Speaker 1");
        assert_eq!(m.label(Some(0)), "Speaker 2");
        m.rename(0, "Jan");
        assert_eq!(m.label(Some(0)), "Jan");
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
