# Phase 1 spike – Nemotron 3 Diarization on Apple Silicon

Measured 2026-10-26 on an **Apple M5 MacBook (macOS 27.0)**, Python 3.13, `transformers 5.20.2`,
`torch 2.14.1` (MPS available), fp32 weights. Scripts in `research/` (see `research/README.md`).
All numbers are measured, not estimated. Timings exclude model load and one warm-up pass.

## Licence

Model card: **OpenMDW License Agreement 1.1** (`license: openmdw-1.1`), not gated, card states
"ready for commercial or non-commercial use". Weights are downloaded on demand into the Hugging Face
cache; nothing is committed. Runtime NeMo-Speech.cpp is Apache-2.0.

## Test material

| File | Kind | Length | Speakers | Reference |
|---|---|---|---|---|
| AMI `ES2004a.Array1-01` | real meeting, single far-field table mic, English | 17.5 min | 4 | pyannote AMI-diarization-setup, `only_words` |
| AMI `IS1009a.Array1-01` | same | 14.0 min | 4 | same |
| `nl_mix_4spk_seed0` | synthetic Dutch, MLS (LibriVox) turns, 10 % short overlaps, light noise | 4.0 min | 4 | generated RTTM |
| `nl_mix_3spk_seed1` | same | 3.0 min | 3 | generated RTTM |

AMI is CC BY 4.0 (downloaded from the Edinburgh mirror); MLS Dutch is CC BY 4.0 over public-domain
LibriVox recordings. `research/make_dutch_mix.py` rebuilds the Dutch files deterministically.
No real recordings of yours were used.

DER = pyannote.metrics, 0.25 s collar, overlap scored, threshold 0.5, gaps < 0.3 s filled.
The synthetic references mark whole turns as speech (including short in-turn pauses), so their
"miss" is inflated; **confusion** is the meaningful number there.

## Offline (30.4 s buffer, whole file in one call)

| File | Device | RTF | DER | miss | FA | confusion | speakers |
|---|---|---|---|---|---|---|---|
| nl_mix_4spk | CPU | 0.0093 | 10.9 % | 10.9 % | 0 | **0 %** | 4/4 |
| ES2004a | CPU | 0.0064 | 32.0 % | 29.2 % | 1.7 % | 1.1 % | 4/4 |
| ES2004a | MPS | 0.0047 | 31.7 % | 28.9 % | 1.7 % | 1.1 % | 4/4 |
| IS1009a | CPU | 0.0065 | 19.4 % | 15.8 % | 2.4 % | 1.1 % | 4/4 |
| IS1009a | MPS | 0.0030 | 19.4 % | 15.8 % | 2.4 % | 1.1 % | 4/4 |

ES2004a is a very quiet recording (RMS 0.004). Peak-normalising to 0.9 lowers its DER to 29.7 %;
loudness normalisation to −23 dBFS made it *worse* (36.4 %, 6 phantom speakers). → at most a
gentle peak gain in the app, no aggressive AGC.

## Streaming (the live path)

Profiles are `(chunk, right-context)` in 80 ms encoder frames. Compute per step is almost constant
(~45 ms MPS, ~115 ms CPU) because every step re-encodes the speaker cache + FIFO (up to 528 frames)
plus the chunk, so **RTF ≈ step time / hop** and shorter hops cost proportionally more.

Speed, IS1009a first 300 s (p80: first 60 s):

| Profile | Buffer | Hop | CPU RTF | CPU step median / p95 | MPS RTF | MPS step median / p95 |
|---|---|---|---|---|---|---|
| p80 (1,0) | 80 ms | 80 ms | **1.20 (not real time)** | 103 / 151 ms | 0.52 | 43 / 63 ms |
| ultra_low (3,1) | 320 ms | 240 ms | 0.49 | 119 / 157 ms | 0.18 | 44 / 54 ms |
| very_low (6,2) | 640 ms | 480 ms | 0.24 | 117 / 158 ms | 0.091 | 44 / 54 ms |
| **low (9,4)** | **1040 ms** | **720 ms** | **0.17** | 115 / 161 ms | **0.066** | 47 / 60 ms |
| p5s (56,8) | 5120 ms | 4480 ms | 0.025 | 112 / 140 ms | 0.012 | 55 / 68 ms |

Accuracy, full files, MPS (CPU gives identical outputs):

| Profile | ES2004a DER (conf.) | IS1009a DER (conf.) | nl 4spk DER (conf.) | nl 3spk DER (conf.) |
|---|---|---|---|---|
| p80 | — | 147 % (FA 131 %, 2/4 speakers) on 60 s | — | — |
| ultra_low | 34.9 % (5.9 %) | 18.6 % (1.4 %) | 12.1 % (0.05 %) | 10.4 % (0 %) |
| very_low | 33.8 % (5.2 %) | 18.7 % (1.4 %) | 10.7 % (0 %) | 10.0 % (0 %) |
| **low** | **30.1 % (2.5 %)** | 18.8 % (1.5 %) | 10.1 % (0 %) | 10.0 % (0 %) |
| p5s | 36.6 % (8.3 %) | 17.9 % (1.2 %) | 9.5 % (0 %) | 10.4 % (0 %) |

All four files: correct speaker count at every profile ≥ 320 ms.

### Findings

- **Real time is easy at ≥ 320 ms**, on CPU and on MPS. 80 ms is both slower than real time on
  CPU and broken (massive false alarms) – consistent with the card's 0.32 s minimum recommendation.
- **Dutch works.** Although Dutch isn't listed in the training data, speaker confusion on the Dutch
  mixes is 0–0.05 % at every profile; diarization is mostly language-independent.
- Main error source on real far-field audio is **missed speech** (backchannels, overlap, quiet
  speakers), not speaker confusion.
- The 5 s profile is not uniformly better: on ES2004a it confused more (8.3 %); `low` was the best
  or near-best everywhere.

**Recommendation: `low_latency` (1.04 s buffer, 0.72 s hop)** as the default; expose
`very_low` / `ultra_low` / 5 s as options. On MPS this uses ~6.5 % of one GPU-second per second.

## Streaming state (what the native port must carry)

`Nemotron3DiarizationSpeakerCache` (transformers) – per stream, batch 1, fp32:

| Tensor | Shape | Meaning |
|---|---|---|
| `embeds` | `(1, 264, 512)` | Arrival-Order Speaker Cache: pre-encoder embeddings of the most informative past frames, grouped by speaker |
| `probs` | `(1, 264, 8)` | speaker probabilities stored with each cache frame (used when re-compressing) |
| `fifo` | `(1, 264, 512)` | FIFO of the most recent pre-encoder embeddings |
| scalars | – | `num_cache_frames`, `num_fifo_frames`, `is_compressed` |

Config (`streaming_config`): `fifo_length 264`, `speaker_cache_length 264`,
`speaker_cache_update_period 222`, `speaker_cache_silence_frames_per_speaker 1`, scores
`prediction_score_threshold 0.25`, `latest_frames_score_boost 0.05`, `strong_boost_rate 0.75`,
`weak_boost_rate 1.5`, `min_positive_scores_rate 0.5`. Offline mode overrides FIFO 40 / update 300.

Per step: features (128-mel, n_fft 512, hop 160, win 400, preemph 0.97; first chunk centred,
later chunks uncentred and overlapping by `n_fft/2`) → feature stacking (8×) → concat
`[cache, fifo, chunk(+lookahead)]` → 31-layer encoder (positions restart each step) → proj →
sub-pixel upsampler → classifier → logits at 10 ms. Then `update()`: chunk frames (not lookahead)
join the FIFO; on overflow ≥ 222 oldest frames move to the cache; when the cache exceeds 264 it is
compressed by a top-k over per-speaker frame scores with boosts and one learned silence embedding
per speaker. The compression is data-dependent control flow (top-k, masking, sort) – it sits
**outside** a single static graph in any export. NeMo-Speech.cpp implements all of this natively.

## Reference artefacts (not committed; regenerate with the scripts)

- `research/out/offline/*.probs.npy` – offline probabilities `(frames, 8)`
- `research/out/streaming/*.chunks.npz` – **per-chunk logits** + audio bounds, per profile/device
- `research/out/streaming/*.rttm|json` – segments + timing + DER
- `research/out/matrix.log` – raw benchmark lines

## App prototype (route C, temporary)

`research/diarize_server.py` exposes the streaming reference over stdin/stdout; the app's
`PythonSidecarDiarizer` drives it. Parity with `diarize_streaming.py` on 60 s of the Dutch mix:
identical frame count (6000), contiguous blocks, max abs probability diff **5.0e-5** (the 4-decimal
rounding of the JSON transport).
