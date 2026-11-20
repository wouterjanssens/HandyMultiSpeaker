//! AI meeting notes from a finished transcript.

use anyhow::Result;

use super::Client;

const INSTRUCTIONS: &str = "You turn a meeting transcript into concise meeting notes.\n\
- Write in the language most of the meeting was held in (usually Dutch or English); keep \
  quotes, product names and jargon as spoken.\n\
- Use the speaker labels exactly as they appear in the transcript; never invent names, \
  numbers, owners or dates.\n\
- Speaker labels and transcription can contain errors: when something is unclear, say so \
  instead of guessing.\n\
- Output Markdown with these sections, leaving out any that would be empty:\n\
  ## Summary (3–6 bullets)\n\
  ## Decisions\n\
  ## Action items (a checklist: `- [ ] owner – task – due date if mentioned`)\n\
  ## Open questions\n\
  ## Topics (one line per topic with who drove it)\n\
- No preamble, no closing remarks.";

/// Markdown notes for `transcript_md` (the saved meeting transcript).
pub fn generate(client: &Client, model: &str, transcript_md: &str) -> Result<String> {
    let notes = client.respond(model, INSTRUCTIONS, transcript_md)?;
    Ok(notes.trim().to_string())
}
