# Stable local code signing

**Problem.** `tauri.conf.json` signs ad hoc (`"signingIdentity": "-"`). An ad-hoc signature is just
the code hash, so every rebuild is a "new app" to macOS and the Microphone and Accessibility
permissions have to be granted again.

**Fix.** Sign every build with the same certificate. macOS then identifies the app by that
certificate, and permissions survive rebuilds. A self-signed certificate is enough for your own Mac
(it is not notarised, so it is not meant for distributing to others).

## One-time setup (≈1 minute)

1. Open **Keychain Access** → menu **Keychain Access → Certificate Assistant → Create a
   Certificate…**
2. Name: `MeetScribe Local Signing` · Identity Type: **Self Signed Root** · Certificate Type:
   **Code Signing** → **Create** → **Continue** / **Done**. It is stored in your _login_ keychain.
3. Check it is visible:

   ```bash
   security find-identity -p codesigning
   ```

   It should list `"MeetScribe Local Signing"`. (It may show as not trusted ("CSSMERR_TP_NOT_TRUSTED");
   that's fine for signing your own builds.)

## Build

```bash
scripts/meetscribe/build_app.sh
```

The script signs with that certificate when it exists (and warns, falling back to ad hoc, when it
doesn't). The last line it prints is the app's designated requirement: with the certificate it
refers to a `certificate … H"…"` hash; ad-hoc builds show only `cdhash H"…"`, which changes every
build.

## After switching

The first build signed with the certificate is still a change of identity, so macOS asks for the
permissions **one last time**. If an old entry gets in the way, remove MeetScribe from **System
Settings → Privacy & Security → Microphone / Accessibility** and grant it again. From then on,
rebuilds keep the permissions.

The first time macOS uses the new key to sign, Keychain may ask whether `codesign` may use it –
choose **Always Allow**.

Name differs? Set `MEETSCRIBE_SIGNING_IDENTITY="Your Certificate Name"` before running the script.
