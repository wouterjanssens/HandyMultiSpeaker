"""Word-level speaker attribution accuracy of an app meeting JSON against a reference RTTM.

Each word's reference speaker = the RTTM speaker active at the word's midpoint (words in
reference gaps or overlaps are skipped). Hypothesis labels are mapped to reference speakers with
the best one-to-one assignment (Hungarian on co-occurrence counts).

Usage: uv run python score_meeting.py meeting.json reference.rttm
"""

import json
import sys
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

meeting, rttm = json.load(open(sys.argv[1])), sys.argv[2]
ref = []
for line in open(rttm):
    p = line.split()
    ref.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))

pairs, skipped = [], 0
for turn in meeting["turns"]:
    for w in turn["words"]:
        mid = (w["start_ms"] + w["end_ms"]) / 2000
        active = {s for a, b, s in ref if a <= mid < b}
        if len(active) != 1 or w["speaker"] is None:
            skipped += 1
            continue
        pairs.append((w["speaker"], active.pop()))

hyp_ids = sorted({h for h, _ in pairs})
ref_ids = sorted({r for _, r in pairs})
counts = Counter(pairs)
m = np.array([[counts[(h, r)] for r in ref_ids] for h in hyp_ids])
rows, cols = linear_sum_assignment(-m)
correct = int(m[rows, cols].sum())
print(json.dumps({
    "words_scored": len(pairs),
    "words_skipped (gap/overlap/unknown)": skipped,
    "speaker_accuracy": round(correct / max(1, len(pairs)), 4),
    "hyp_speakers": len(hyp_ids),
    "ref_speakers": len(ref_ids),
    "turns": len(meeting["turns"]),
}, indent=1))
