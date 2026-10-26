"""Offline Nemotron 3 Diarization over a WAV: writes RTTM + JSON segments (+ DER if a reference is given).

Offline mode = the forward chunks the whole recording itself with the config's offline sizes
(chunk 340 / right context 40 encoder frames = 30.4 s buffer, FIFO 40).

Usage:
  uv run python diarize_offline.py data/ami/ES2004a.Array1-01.wav --ref data/ami/ES2004a.rttm [--device mps]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import SR, evaluate, load_audio, load_model, pick_device, probs_to_segments, write_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ref", help="reference RTTM for DER")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="out/offline")
    args = ap.parse_args()

    device = pick_device(args.device)
    processor, model = load_model(device)
    audio = load_audio(args.wav, args.max_seconds)
    duration = len(audio) / SR

    inputs = processor(audio, sampling_rate=SR).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    probs = logits.sigmoid()[0].float().cpu().numpy()

    uri = Path(args.wav).stem
    out = Path(args.out_dir) / f"{uri}.{device}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out}.probs.npy", probs)
    segments = probs_to_segments(probs, args.threshold, args.threshold)
    result = {
        "device": device,
        "audio_s": round(duration, 2),
        "compute_s": round(elapsed, 3),
        "rtf": round(elapsed / duration, 4),
        "num_frames": int(probs.shape[0]),
    }
    if args.ref:
        result["eval"] = evaluate(Path(args.ref), segments, uem_end=duration)
    write_outputs(out, uri, segments, extra=result)
    print(result)


if __name__ == "__main__":
    main()
