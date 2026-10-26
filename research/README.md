# research/ – Phase 1 reference implementation (not part of the app build)

Python ground truth for Nemotron 3 Diarization, plus the temporary diarizer sidecar the app uses
until the native backend lands. Results: `docs/meetscribe/SPIKE-RESULTS.md`.

```bash
cd research
uv sync                                   # Python 3.13 env (transformers ≥ 5.18, torch, …)
uv run python make_dutch_mix.py           # synthetic Dutch test meeting → data/synthetic/
uv run python diarize_offline.py   data/ami/IS1009a.Array1-01.wav --ref data/ami/IS1009a.rttm --device mps
uv run python diarize_streaming.py data/ami/IS1009a.Array1-01.wav --ref data/ami/IS1009a.rttm --profile low --device cpu
./run_matrix.sh                           # full benchmark → out/matrix.log
```

Test audio (not committed; `data/` is git-ignored):

```bash
mkdir -p data/ami && cd data/ami
for m in ES2004a IS1009a; do
  curl -fLO https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Array1-01.wav
  curl -fLo $m.rttm https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/$m.rttm
done
mkdir -p ../mls_nl && curl -fLo ../mls_nl/1_hours.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/dutch/1_hours-00000-of-00001.parquet
```

| File | Purpose |
|---|---|
| `common.py` | audio loading, model loading, hysteresis segmentation, RTTM/JSON output, DER |
| `diarize_offline.py` | whole-file pass (30.4 s buffer) |
| `diarize_streaming.py` | chunk-by-chunk streaming with the speaker cache; saves per-chunk logits |
| `diarize_server.py` | **temporary** stdin/stdout sidecar used by the app (`meeting/sidecar.rs`) |
| `make_dutch_mix.py` | builds Dutch multi-speaker test files with ground truth |
| `run_matrix.sh` | latency-profile × device benchmark |

Testing the app without a microphone: start the dev app with
`MEETSCRIBE_SIMULATE_WAV=/abs/path.wav [MEETSCRIBE_SIMULATE_SPEED=1] [MEETSCRIBE_SIMULATE_AUTOSTOP=1]`
and toggle a meeting (`handy --toggle-meeting`); the file replaces the microphone in real time.
