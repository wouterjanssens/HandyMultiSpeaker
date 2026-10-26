"""Build a synthetic Dutch multi-speaker "meeting" with a ground-truth RTTM.

Source: Multilingual LibriSpeech, Dutch, `1_hours` split (facebook/multilingual_librispeech on the
Hugging Face Hub; LibriVox public-domain audiobook recordings, dataset under CC BY 4.0).

Turns are pieces of single-speaker utterances (cut at the quietest 20 ms near the target length so
words are rarely split), separated by short gaps, with occasional short overlaps.

Usage: uv run python make_dutch_mix.py [--speakers 4] [--minutes 4] [--seed 0]
"""

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
HERE = Path(__file__).parent


def load_utterances(path: Path, num_speakers: int):
    table = pq.read_table(path)
    rows = table.to_pylist()
    by_speaker: dict[str, list[dict]] = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    chosen = sorted(by_speaker, key=lambda s: -len(by_speaker[s]))[:num_speakers]
    out = {}
    for spk in chosen:
        clips = []
        for row in by_speaker[spk]:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = resample_poly(audio, SR, sr).astype(np.float32)
            clips.append((audio, row["transcript"]))
        out[spk] = clips
    return out


def quiet_cut(audio: np.ndarray, target: int, search: int = SR // 2) -> int:
    """Sample index of the quietest 20 ms window within +-search of target."""
    win = SR // 50
    lo, hi = max(win, target - search), min(len(audio) - win, target + search)
    if hi <= lo:
        return min(target, len(audio))
    energies = [float(np.mean(audio[i : i + win] ** 2)) for i in range(lo, hi, win // 2)]
    return lo + int(np.argmin(energies)) * (win // 2) + win // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap-prob", type=float, default=0.1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    utts = load_utterances(HERE / "data/mls_nl/1_hours.parquet", args.speakers)
    speakers = list(utts)
    pools = {s: list(range(len(utts[s]))) for s in speakers}
    for s in speakers:
        rng.shuffle(pools[s])
    cursors = {s: (None, 0) for s in speakers}  # (utterance index, sample offset)

    total = int(args.minutes * 60 * SR)
    mix = np.zeros(total + 30 * SR, dtype=np.float32)
    segments = []
    t = int(0.5 * SR)
    prev = None
    while t < total:
        spk = rng.choice([s for s in speakers if s != prev])
        prev = spk
        idx, off = cursors[spk]
        if idx is None or off >= len(utts[spk][idx][0]) - SR:
            if not pools[spk]:
                pools[spk] = list(range(len(utts[spk])))
                rng.shuffle(pools[spk])
            idx, off = pools[spk].pop(), 0
        audio = utts[spk][idx][0]
        want = int(rng.uniform(1.5, 10.0) * SR)
        end = quiet_cut(audio, off + want) if off + want < len(audio) - SR // 2 else len(audio)
        piece = audio[off:end]
        cursors[spk] = (idx, end)
        if len(piece) < SR // 2:
            continue
        # trim leading/trailing near-silence so the reference only covers speech
        level = np.abs(piece) > 0.01
        if level.any():
            first, last = np.argmax(level), len(level) - np.argmax(level[::-1])
            piece = piece[max(0, first - 800) : min(len(piece), last + 800)]
        piece = piece / (np.max(np.abs(piece)) + 1e-9) * rng.uniform(0.3, 0.6)
        mix[t : t + len(piece)] += piece
        segments.append((t / SR, len(piece) / SR, spk))
        if rng.random() < args.overlap_prob:
            gap = -int(rng.uniform(0.3, 1.0) * SR)
        else:
            gap = int(rng.uniform(0.15, 1.2) * SR)
        t = max(t + len(piece) + gap, t + SR // 2)

    end = int(max(s + d for s, d, _ in segments) * SR) + SR // 2
    mix = mix[:end]
    # mild room noise so silence isn't digital zero
    mix += np.random.default_rng(args.seed).normal(0, 0.002, len(mix)).astype(np.float32)
    mix = np.clip(mix, -1, 1)

    name = f"nl_mix_{args.speakers}spk_seed{args.seed}"
    out_dir = HERE / "data/synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / f"{name}.wav", mix, SR, subtype="PCM_16")
    with open(out_dir / f"{name}.rttm", "w") as f:
        for start, dur, spk in segments:
            f.write(f"SPEAKER {name} 1 {start:.3f} {dur:.3f} <NA> <NA> nl{spk} <NA> <NA>\n")
    with open(out_dir / f"{name}.uem", "w") as f:
        f.write(f"{name} 1 0.000 {len(mix) / SR:.3f}\n")
    meta = {
        "source": "facebook/multilingual_librispeech dutch/1_hours (LibriVox, CC BY 4.0)",
        "speakers": speakers,
        "duration_s": len(mix) / SR,
        "num_turns": len(segments),
        "seed": args.seed,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
