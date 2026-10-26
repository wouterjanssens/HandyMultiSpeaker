"""Streaming diarizer sidecar (TEMPORARY prototype backend for the app, handoff route C).

Runs the Hugging Face Nemotron 3 Diarization streaming reference (same code path as
diarize_streaming.py) behind a tiny stdin/stdout protocol so the Rust app can use it until the
native NeMo-Speech.cpp backend is available. Audio never leaves the machine.

Protocol
  stdin : repeated [u32 LE n][n x f32 LE] blocks of 16 kHz mono audio; n == 0 means "finish".
  stdout: JSON lines.
    first line: {"type":"ready","num_speakers":8,"frame_ms":10,"latency_ms":1040,"describe":"..."}
    per step  : {"type":"frames","start_frame":k,"num_frames":m,"probs":[m*8 floats],"finalized_until":k+m}
    at the end: {"type":"done"}
  errors go to stderr; the process exits non-zero.

Usage: uv run --project research python research/diarize_server.py --profile low --device mps
"""

import argparse
import json
import struct
import sys

import numpy as np
import torch

from common import SR, load_model, pick_device
from diarize_streaming import ALIASES, PROFILES


def send(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def read_block(stream) -> np.ndarray | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("<I", header)
    if n == 0:
        return None
    data = stream.read(4 * n)
    if len(data) < 4 * n:
        return None
    return np.frombuffer(data, dtype="<f4").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="low_latency")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    profile = ALIASES.get(args.profile, args.profile)
    device = pick_device(args.device)
    processor, model = load_model(device)
    processor.streaming_modes = {**processor.streaming_modes, **PROFILES}
    processor.set_streaming_mode(profile)
    num_speakers = model.config.head_config.num_speakers

    n_first = processor.num_samples_first_audio_chunk
    n_rest = processor.num_samples_per_audio_chunk
    step = processor.num_mel_frames_per_step

    send(
        {
            "type": "ready",
            "num_speakers": num_speakers,
            "frame_ms": 10,
            "latency_ms": processor.streaming_latency_ms,
            "describe": f"nvidia/Nemotron-3-Diarization (transformers, {profile}, {device}) via Python sidecar",
        }
    )

    buf = np.zeros(0, dtype=np.float32)  # audio from absolute sample `buf_start`
    buf_start = 0
    cache = None
    k = 0
    stdin = sys.stdin.buffer

    def run_chunk(chunk: np.ndarray, first: bool, last: bool):
        nonlocal cache
        inputs = processor(
            chunk, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=first, is_last_audio_chunk=last
        ).to(device)
        with torch.inference_mode():
            out = model(**inputs, speaker_cache=cache)
        cache = out.speaker_cache
        return torch.sigmoid(out.logits[0]).float().cpu().numpy()

    def emit(start_frame: int, probs: np.ndarray):
        send(
            {
                "type": "frames",
                "start_frame": start_frame,
                "num_frames": int(probs.shape[0]),
                "probs": [round(float(x), 4) for x in probs.reshape(-1)],
                "finalized_until": start_frame + int(probs.shape[0]),
            }
        )

    finished = False
    while not finished:
        block = read_block(stdin)
        if block is None:
            finished = True
        else:
            buf = np.concatenate([buf, block])
        # process every complete chunk
        while True:
            if k == 0:
                start, need = 0, n_first
            else:
                start, need = processor.audio_chunk_start(k * step), n_rest
            end = start + need
            have_until = buf_start + len(buf)
            if end > have_until:
                break
            chunk = buf[start - buf_start : end - buf_start]
            emit(k * step, run_chunk(chunk, k == 0, False))
            k += 1
            # drop audio no future chunk needs
            next_start = processor.audio_chunk_start(k * step)
            if next_start > buf_start:
                buf = buf[next_start - buf_start :]
                buf_start = next_start
        if finished:
            start = 0 if k == 0 else processor.audio_chunk_start(k * step)
            tail = buf[max(0, start - buf_start) :]
            if len(tail) > 0:
                emit(k * step, run_chunk(tail, k == 0, True))
            send({"type": "done"})


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
