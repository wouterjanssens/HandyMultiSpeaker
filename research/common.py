"""Shared helpers for the Phase 1 reference scripts."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_ID = "nvidia/Nemotron-3-Diarization"
SR = 16000
FRAME_S = 0.01  # one logit frame per 10 ms (spectrogram hop)


def load_audio(path: str | Path, max_seconds: float | None = None) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = resample_poly(audio, SR, sr).astype(np.float32)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * SR)]
    return audio


def pick_device(name: str):
    import torch

    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def load_model(device: str):
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.to(device).eval()
    return processor, model


def probs_to_segments(
    probs: np.ndarray, onset: float = 0.5, offset: float = 0.5, min_on_s: float = 0.0, min_off_s: float = 0.0
) -> list[dict]:
    """Hysteresis thresholding of (num_frames, num_speakers) probabilities into segments."""
    segments = []
    num_frames, num_speakers = probs.shape
    for spk in range(num_speakers):
        active, start = False, 0
        spans = []
        for i in range(num_frames):
            p = probs[i, spk]
            if not active and p > onset:
                active, start = True, i
            elif active and p < offset:
                active = False
                spans.append([start, i])
        if active:
            spans.append([start, num_frames])
        # fill short gaps, then drop short segments
        merged = []
        for s, e in spans:
            if merged and (s - merged[-1][1]) * FRAME_S < min_off_s:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if (e - s) * FRAME_S >= min_on_s:
                segments.append({"start": round(s * FRAME_S, 2), "end": round(e * FRAME_S, 2), "speaker": spk})
    segments.sort(key=lambda x: (x["start"], x["speaker"]))
    return segments


def write_outputs(out_base: Path, uri: str, segments: list[dict], extra: dict | None = None):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_base}.rttm", "w") as f:
        for s in segments:
            f.write(
                f"SPEAKER {uri} 1 {s['start']:.2f} {s['end'] - s['start']:.2f} <NA> <NA> spk{s['speaker']} <NA> <NA>\n"
            )
    payload = {"uri": uri, "segments": segments}
    if extra:
        payload.update(extra)
    Path(f"{out_base}.json").write_text(json.dumps(payload, indent=2))


def load_rttm(path: Path):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        ann[Segment(start, start + dur)] = spk
    return ann


def evaluate(ref_rttm: Path, hyp_segments: list[dict], uem_end: float | None = None, collar: float = 0.25) -> dict:
    """DER (pyannote.metrics) with a collar, overlaps scored; plus speaker counts."""
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = load_rttm(ref_rttm)
    hyp = Annotation()
    for s in hyp_segments:
        if s["end"] > s["start"]:
            hyp[Segment(s["start"], s["end"])] = f"spk{s['speaker']}"
    if uem_end is None:
        uem_end = max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end if hyp else 0)
    uem = Timeline([Segment(0, uem_end)])
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    detail = metric(ref, hyp, uem=uem, detailed=True)
    total = detail["total"] or 1.0
    return {
        "DER": round(detail["diarization error rate"], 4),
        "miss": round(detail["missed detection"] / total, 4),
        "false_alarm": round(detail["false alarm"] / total, 4),
        "confusion": round(detail["confusion"] / total, 4),
        "ref_speakers": len(ref.labels()),
        "hyp_speakers": len(hyp.labels()),
        "collar": collar,
    }
