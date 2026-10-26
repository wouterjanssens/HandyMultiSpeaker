#!/bin/sh
# Phase 1 benchmark matrix. Results land in out/streaming/*.json; summary in out/matrix.log.
set -e
cd "$(dirname "$0")"
log=out/matrix.log; mkdir -p out; : > $log
run() { echo "== $*" >> $log; uv run python diarize_streaming.py "$@" 2>/dev/null | grep '^{' >> $log; }
# speed: all profiles x devices on the first 300 s of IS1009a
for d in cpu mps; do
  run data/ami/IS1009a.Array1-01.wav --ref data/ami/IS1009a.rttm --profile p80 --device $d --max-seconds 60 --out-dir out/streaming_speed
  for p in ultra very_low low p5s; do
    run data/ami/IS1009a.Array1-01.wav --ref data/ami/IS1009a.rttm --profile $p --device $d --max-seconds 300 --out-dir out/streaming_speed
  done
done
# accuracy: full files, MPS, three main profiles
for f in ami/ES2004a ami/IS1009a synthetic/nl_mix_4spk_seed0 synthetic/nl_mix_3spk_seed1; do
  w=data/$f.wav; [ -f $w ] || w=data/$f.Array1-01.wav
  for p in ultra very_low low p5s; do run $w --ref data/$f.rttm --profile $p --device mps; done
done
echo DONE >> $log
