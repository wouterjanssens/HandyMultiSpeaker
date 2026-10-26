"""Parity check: native NeMo-Speech.cpp frame probabilities vs the Python reference.

  diarize-cli data/synthetic/nl_mix_4spk_seed0.wav --model <q8_0.gguf> --dump-probs out/native.f32 > out/native.rttm
  uv run python diarize_streaming.py data/synthetic/nl_mix_4spk_seed0.wav --profile low --device cpu
  uv run python compare_native.py out/native.f32 out/streaming/nl_mix_4spk_seed0.low_latency.cpu.probs.npy \
      --rttm out/native.rttm --ref data/synthetic/nl_mix_4spk_seed0.rttm

Note the native model is q8_0 and the reference fp32, so expect small differences; the decisive
checks are the decision agreement (p > 0.5) and the DER of each against the ground truth.
"""

import argparse
from pathlib import Path

import numpy as np

from common import evaluate, probs_to_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native_f32")
    ap.add_argument("reference_npy")
    ap.add_argument("--rttm", help="native RTTM from diarize-cli")
    ap.add_argument("--ref", help="ground-truth RTTM")
    args = ap.parse_args()

    ref = np.load(args.reference_npy)
    nat = np.fromfile(args.native_f32, dtype="<f4").reshape(-1, ref.shape[1])
    n = min(len(ref), len(nat))
    diff = np.abs(nat[:n] - ref[:n])
    agree = ((nat[:n] > 0.5) == (ref[:n] > 0.5)).mean()
    print(f"frames native {len(nat)} reference {len(ref)}")
    print(f"max abs diff {diff.max():.4f}  mean abs diff {diff.mean():.5f}  p99 {np.percentile(diff, 99):.4f}")
    print(f"per-frame/speaker decision agreement (p>0.5): {agree * 100:.2f} %")
    if args.ref:
        dur = n / 100
        print("DER reference (fp32 python):", evaluate(Path(args.ref), probs_to_segments(ref, 0.5, 0.5, 0, 0.3), dur))
        print("DER native probs (same postproc):", evaluate(Path(args.ref), probs_to_segments(nat, 0.5, 0.5, 0, 0.3), dur))


if __name__ == "__main__":
    main()
