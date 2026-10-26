"""Generate MeetScribe placeholder icons (app icons + tray icons).

Original artwork for the MeetScribe fork: two overlapping speech bubbles.
It deliberately replaces Handy's brand assets (hand / glove), which are not
covered by Handy's MIT licence.

Run from the repo root:

    uv run --with pillow --with resvg-py python scripts/meetscribe/make_icons.py

Writes every file in src-tauri/icons/ (same names and pixel sizes; icon.icns
via `iconutil`, icon.ico via Pillow) and the tray_*.png / colored tray icons in
src-tauri/resources/. The tray geometry matches
src/components/icons/TrayIcon.tsx and HandyHand.tsx.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import resvg_py
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ICONS = ROOT / "src-tauri" / "icons"
RESOURCES = ROOT / "src-tauri" / "resources"

INDIGO = "#4F46E5"
TEAL = "#0EA5A4"
# Colored (Linux) tray icon color.
TRAY_COLOR = "#6366F1"


def render(svg: str, size: int) -> Image.Image:
    png = resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size)
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


# --------------------------------------------------------------------------
# App icon (1024 grid)
# --------------------------------------------------------------------------

BUBBLES_1024 = f"""
  <g>
    <rect x="400" y="230" width="420" height="320" rx="110" fill="#fff" fill-opacity="0.55"/>
    <path d="M700 540 L790 540 L800 650 Z" fill="#fff" fill-opacity="0.55"/>
  </g>
  <g>
    <rect x="204" y="410" width="440" height="330" rx="115" fill="#fff"/>
    <path d="M262 720 L372 720 L232 830 Z" fill="#fff"/>
    <rect x="280" y="502" width="290" height="38" rx="19" fill="{INDIGO}"/>
    <rect x="280" y="566" width="240" height="38" rx="19" fill="{INDIGO}"/>
    <rect x="280" y="630" width="170" height="38" rx="19" fill="{INDIGO}"/>
  </g>
"""

GRADIENT = f"""
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{INDIGO}"/>
      <stop offset="1" stop-color="{TEAL}"/>
    </linearGradient>
  </defs>
"""


def app_icon_svg(margin: float, radius: float, shape: str = "squircle") -> str:
    """Rounded-square (or circle / full-bleed) background with the bubbles."""
    size = 1024 - 2 * margin
    if shape == "circle":
        bg = f'<circle cx="512" cy="512" r="{size / 2}" fill="url(#bg)"/>'
    else:
        bg = (
            f'<rect x="{margin}" y="{margin}" width="{size}" height="{size}" '
            f'rx="{radius}" fill="url(#bg)"/>'
        )
    # Shrink the artwork together with the background.
    scale = size / 1024
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">'
        f"{GRADIENT}{bg}"
        f'<g transform="translate({margin} {margin}) scale({scale})">{BUBBLES_1024}</g>'
        "</svg>"
    )


def foreground_svg() -> str:
    """Android adaptive-icon foreground: bubbles only, inside the safe zone."""
    # Safe zone is the central 66/108 of the canvas.
    scale = 66 / 108
    off = 1024 * (1 - scale) / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">'
        f'<g transform="translate({off} {off}) scale({scale})">{BUBBLES_1024}</g>'
        "</svg>"
    )


MAC = app_icon_svg(margin=100, radius=185)  # macOS icon grid (824 + padding)
DESKTOP = app_icon_svg(margin=24, radius=210)  # Windows / Linux / Store tiles
FULL_BLEED = app_icon_svg(margin=0, radius=0)  # iOS (system applies the mask)
ROUND = app_icon_svg(margin=24, radius=0, shape="circle")


# --------------------------------------------------------------------------
# Tray icons (64 grid). Template style: one flat color on transparency.
# --------------------------------------------------------------------------

# Front bubble expanded by 3px: cut out of the back bubble for a clean gap.
FRONT_GAP = '<rect x="3" y="19" width="44" height="35" rx="11"/><path d="M8 44 L28 44 L6 63 Z"/>'


def bubbles(color: str, mask_id: str, extra_cut: str = "") -> str:
    return f"""
  <mask id="{mask_id}">
    <rect width="64" height="64" fill="#fff"/>
    <g fill="#000">{FRONT_GAP}{extra_cut}</g>
  </mask>
  <g mask="url(#{mask_id})">
    <rect x="24.5" y="8.5" width="33" height="24" rx="8" fill="none" stroke="{color}" stroke-width="5"/>
    <path d="M43 33 L53 33 L55 42 Z" fill="{color}"/>
  </g>
  <g{f' mask="url(#{mask_id}b)"' if extra_cut else ""}>
    <rect x="8.5" y="24.5" width="33" height="24" rx="8" fill="none" stroke="{color}" stroke-width="5"/>
    <path d="M12 49 L23 49 L10 59 Z" fill="{color}"/>
  </g>
"""


def tray_svg(state: str, color: str) -> str:
    head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    if state == "idle":
        body = bubbles(color, "gap")
    elif state == "warning":
        badge_gap = '<circle cx="47" cy="47" r="17"/>'
        body = f"""
  <mask id="gapb">
    <rect width="64" height="64" fill="#fff"/>
    <g fill="#000">{badge_gap}</g>
  </mask>
  {bubbles(color, "gap", badge_gap)}
  <mask id="bang">
    <rect width="64" height="64" fill="#fff"/>
    <rect x="45" y="36" width="4" height="13" rx="2" fill="#000"/>
    <circle cx="47" cy="54.5" r="2.4" fill="#000"/>
  </mask>
  <circle cx="47" cy="47" r="14" fill="{color}" mask="url(#bang)"/>
"""
    elif state == "recording":
        # Filled disc with a level-meter / waveform cut out.
        bars = "".join(
            f'<rect x="{x - 2.5}" y="{32 - h / 2}" width="5" height="{h}" rx="2.5" fill="#000"/>'
            for x, h in ((17, 12), (24.5, 24), (32, 34), (39.5, 24), (47, 12))
        )
        body = f"""
  <mask id="cut"><rect width="64" height="64" fill="#fff"/>{bars}</mask>
  <circle cx="32" cy="32" r="31" fill="{color}" mask="url(#cut)"/>
"""
    elif state == "transcribing":
        # Filled disc with lines of text cut out.
        lines = "".join(
            f'<rect x="15" y="{y}" width="{w}" height="5.5" rx="2.75" fill="#000"/>'
            for y, w in ((19, 34), (29.25, 28), (39.5, 19))
        )
        body = f"""
  <mask id="cut"><rect width="64" height="64" fill="#fff"/>{lines}</mask>
  <circle cx="32" cy="32" r="31" fill="{color}" mask="url(#cut)"/>
"""
    else:
        raise ValueError(state)
    return head + body + "</svg>"


# --------------------------------------------------------------------------


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def app_icons() -> None:
    desktop_pngs = {
        "32x32.png": 32,
        "64x64.png": 64,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "icon.png": 512,
        "logo.png": 1024,
        "Square30x30Logo.png": 30,
        "Square44x44Logo.png": 44,
        "Square71x71Logo.png": 71,
        "Square89x89Logo.png": 89,
        "Square107x107Logo.png": 107,
        "Square142x142Logo.png": 142,
        "Square150x150Logo.png": 150,
        "Square284x284Logo.png": 284,
        "Square310x310Logo.png": 310,
        "StoreLogo.png": 50,
    }
    for name, size in desktop_pngs.items():
        save(render(DESKTOP, size), ICONS / name)

    # Windows .ico (same sizes as before).
    ico_sizes = [16, 24, 32, 48, 64, 256]
    big = render(DESKTOP, 256)
    big.save(
        ICONS / "icon.ico",
        format="ICO",
        sizes=[(s, s) for s in ico_sizes],
        append_images=[render(DESKTOP, s) for s in ico_sizes],
    )

    # macOS .icns via iconutil.
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            save(render(MAC, base), iconset / f"icon_{base}x{base}.png")
            save(render(MAC, base * 2), iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(ICONS / "icon.icns")],
            check=True,
        )

    # iOS: opaque, full-bleed squares.
    for path in sorted((ICONS / "ios").glob("*.png")):
        size = Image.open(path).size[0]
        img = render(FULL_BLEED, size)
        opaque = Image.new("RGBA", img.size, INDIGO)
        opaque.alpha_composite(img)
        save(opaque, path)

    # Android launcher icons.
    for path in sorted((ICONS / "android").glob("*/*.png")):
        size = Image.open(path).size[0]
        if path.name == "ic_launcher_foreground.png":
            svg = foreground_svg()
        elif path.name == "ic_launcher_round.png":
            svg = ROUND
        else:
            svg = DESKTOP
        save(render(svg, size), path)


def tray_icons() -> None:
    files = {
        "idle": ("tray_idle", "handy"),
        "warning": ("tray_idle_warning", "handy_warning"),
        "recording": ("tray_recording", "recording"),
        "transcribing": ("tray_transcribing", "transcribing"),
    }
    for state, (mono, colored) in files.items():
        # `tray_*` is white (for dark menu bars), `tray_*_dark` is black.
        save(render(tray_svg(state, "#FFFFFF"), 64), RESOURCES / f"{mono}.png")
        save(render(tray_svg(state, "#000000"), 64), RESOURCES / f"{mono}_dark.png")
        save(render(tray_svg(state, TRAY_COLOR), 64), RESOURCES / f"{colored}.png")


def main() -> None:
    if shutil.which("iconutil") is None:
        raise SystemExit("iconutil not found (macOS only)")
    app_icons()
    tray_icons()
    print("icons written")


if __name__ == "__main__":
    main()
