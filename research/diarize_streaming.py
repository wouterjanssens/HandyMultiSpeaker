"""Streaming Nemotron 3 Diarization, fed chunk by chunk the way a live mic would.

Uses the processor's streaming API (`is_streaming`, `is_first_audio_chunk`, `is_last_audio_chunk`)
and carries `Nemotron3DiarizationSpeakerCache` (AOSC + FIFO) across forwards.

Latency profiles are (chunk_length, chunk_right_context) in 80 ms encoder frames. The checkpoint's
named modes are ultra_low (3,1)=320 ms, very_low (6,2)=640 ms, low (9,4)=1040 ms; we add
p80 (1,0)=80 ms (below the model card's 0.32 s recommendation) and p5s (56,8)=5.12 s.

Outputs (out/streaming/<uri>.<profile>.<device>.*):
  .rttm / .json  segments + timing stats (+ DER with --ref)
  .chunks.npz    per-chunk logits (exact numerical references for the native port)
  .probs.npy     concatenated sigmoid probabilities, (num_frames, 8), 10 ms frames

Usage:
  uv run python diarize_streaming.py data/ami/IS1009a.Array1-01.wav --ref data/ami/IS1009a.rttm \
      --profile low --device cpu
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import SR, evaluate, load_audio, load_model, pick_device, probs_to_segments, write_outputs

PROFILES = {
    "p80": (1, 0),
    "ultra_low_latency": (3, 1),
    "very_low_latency": (6, 2),
    "low_latency": (9, 4),
    "p5s": (56, 8),
}
ALIASES = {"ultra": "ultra_low_latency", "very_low": "very_low_latency", "low": "low_latency"}


def stream(processor, model, audio: np.ndarray, device: str, sync):
    """Yields (chunk_index, audio_start, audio_end, logits[np], compute_seconds, cache) per step."""
    n_first = processor.num_samples_first_audio_chunk
    n_rest = processor.num_samples_per_audio_chunk
    step = processor.num_mel_frames_per_step
    cache = None
    k = 0
    while True:
        if k == 0:
            start, end = 0, n_first
        else:
            start = processor.audio_chunk_start(k * step)
            end = start + n_rest
        is_last = end >= len(audio)
        chunk = audio[start : min(end, len(audio))]
        inputs = processor(
            chunk, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=(k == 0), is_last_audio_chunk=is_last
        ).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs, speaker_cache=cache)
        sync()
        dt = time.perf_counter() - t0
        cache = out.speaker_cache
        yield k, start, min(end, len(audio)), out.logits[0].float().cpu().numpy(), dt, cache
        if is_last:
            return
        k += 1


def describe_cache(cache) -> dict:
    def t(x):
        return None if x is None else {"shape": list(x.shape), "dtype": str(x.dtype)}

    return {
        "embeds (speaker cache, AOSC)": t(cache.embeds),
        "probs (per-cache-frame speaker probs)": t(cache.probs),
        "fifo (recent encoder frames)": t(cache.fifo),
        "num_cache_frames": cache.num_cache_frames,
        "num_fifo_frames": cache.num_fifo_frames,
        "is_compressed": cache.is_compressed,
        "fifo_length": cache.fifo_length,
        "speaker_cache_length": cache.speaker_cache_length,
        "speaker_cache_update_period": cache.speaker_cache_update_period,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ref")
    ap.add_argument("--profile", default="low_latency")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="out/streaming")
    args = ap.parse_args()

    profile = ALIASES.get(args.profile, args.profile)
    device = pick_device(args.device)
    processor, model = load_model(device)
    processor.streaming_modes = {**processor.streaming_modes, **PROFILES}
    processor.set_streaming_mode(profile)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    audio = load_audio(args.wav, args.max_seconds)
    duration = len(audio) / SR

    # warm-up (MPS kernel compilation) on 2 s of silence; not timed
    for _ in stream(processor, model, np.zeros(2 * SR, np.float32), device, sync):
        pass

    chunk_logits, times, bounds = [], [], []
    cache = None
    for k, start, end, logits, dt, cache in stream(processor, model, audio, device, sync):
        chunk_logits.append(logits)
        times.append(dt)
        bounds.append((start, end))
    logits = np.concatenate(chunk_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))

    uri = Path(args.wav).stem
    out = Path(args.out_dir) / f"{uri}.{profile}.{device}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out}.probs.npy", probs)
    np.savez_compressed(
        f"{out}.chunks.npz",
        **{f"chunk_{i:05d}": x for i, x in enumerate(chunk_logits)},
        bounds=np.array(bounds),
    )

    hop_s = processor.num_mel_frames_per_step * 0.01
    times_arr = np.array(times[1:-1] or times)
    result = {
        "device": device,
        "profile": profile,
        "streaming_latency_ms": processor.streaming_latency_ms,
        "chunk_hop_ms": round(hop_s * 1000),
        "num_chunks": len(times),
        "audio_s": round(duration, 2),
        "compute_s": round(float(np.sum(times)), 3),
        "rtf": round(float(np.sum(times)) / duration, 4),
        "chunk_ms_median": round(float(np.median(times_arr)) * 1000, 2),
        "chunk_ms_p95": round(float(np.percentile(times_arr, 95)) * 1000, 2),
        "chunk_ms_max": round(float(np.max(times_arr)) * 1000, 2),
        "num_frames": int(probs.shape[0]),
        "final_state": describe_cache(cache),
    }
    segments = probs_to_segments(probs, args.threshold, args.threshold, 0.0, 0.3)
    if args.ref:
        result["eval"] = evaluate(Path(args.ref), segments, uem_end=duration)
    write_outputs(out, uri, segments, extra=result)
    print(json.dumps({k: v for k, v in result.items() if k != "final_state"}))


if __name__ == "__main__":
    main()
