#!/usr/bin/env bash
# Builds the release MeetScribe.app, signed with a stable local identity so macOS keeps the
# Microphone and Accessibility permissions across rebuilds.
#
# Ad-hoc signing (tauri.conf.json's "-") gives every build a new code hash, and macOS ties
# privacy permissions to that hash - so each rebuild looked like a new app. Signing with the
# same certificate every time makes the app's identity the certificate instead.
#
# One-time setup (Keychain Access, ~1 minute): see docs/meetscribe/SIGNING.md.
# Identity name defaults to "MeetScribe Local Signing"; override with MEETSCRIBE_SIGNING_IDENTITY.
#
# Usage: scripts/meetscribe/build_app.sh
set -euo pipefail

IDENTITY="${MEETSCRIBE_SIGNING_IDENTITY:-MeetScribe Local Signing}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$ROOT/src-tauri/target/release/bundle/macos/MeetScribe.app"

if [ -d /opt/homebrew/opt/rustup/bin ]; then
  export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
fi

# `find-identity` without -v also lists self-signed certificates that aren't trusted
if security find-identity -p codesigning | grep -q "\"$IDENTITY\""; then
  export APPLE_SIGNING_IDENTITY="$IDENTITY"
  echo "Signing with \"$IDENTITY\" (permissions survive rebuilds)"
else
  echo "warning: code-signing identity \"$IDENTITY\" not found - falling back to ad-hoc signing;" >&2
  echo "         macOS will ask for Microphone/Accessibility again. See docs/meetscribe/SIGNING.md" >&2
fi

cd "$ROOT"
bun run tauri build --bundles app

echo
codesign -dr - "$APP" 2>&1 | tail -1
echo "Built: $APP"
