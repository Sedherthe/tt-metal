"""The step sweep, how far the audio moves: whole-utterance log-mel L1 (the stage A gate's own measure,
`tests/pcc/test_hift_chunked.py:_logmel`) between wavs of the same tokens and the same noise draw.

- k steps against 10 steps, each side against itself (TT: the sweep against D43's draws; the reference likewise),
  Stage 1 and streaming;
- for scale:
  - two noise draws of the same side and step count against each other (seed 1 against seeds 2-5), which is what a
    different vocoder noise realization alone moves;
  - TT against upstream at the same step count, streaming only (both stream TT's tokens, so the wavs align; Stage 1's
    tokens differ between the sides). The two draw their vocoder noise differently (TT's own generator, upstream's
    `seed x 1000 + n` per call), so this includes a noise-realization difference as well.

Per cell: the mean and the largest over the six utterances and five draws.

    /opt/venv/bin/python steps_distance.py > steps_distance.md    # host only
"""
import os

import numpy as np
import soundfile
import torch

from models.experimental.cosyvoice2.tests.pcc.test_hift_chunked import _logmel

STEPS = (10, 8, 6, 5)
SEEDS = (1, 2, 3, 4, 5)
D = "/home/user/data/cosyvoice2_steps"
D43 = "/home/user/data/cosyvoice2_draws"


def run_dir(side, mode, k, seed):
    return os.path.join(D43 if k == 10 else os.path.join(D, f"steps{k}"), f"{side}_{mode}_seed{seed}")


def wavs(path):
    return {f: soundfile.read(os.path.join(path, f), dtype="float32")[0] for f in sorted(os.listdir(path)) if f.endswith(".wav")}


cache = {}


def logmel(side, mode, k, seed):
    key = (side, mode, k, seed)
    if key not in cache:
        cache[key] = {f: _logmel(torch.from_numpy(w)) for f, w in wavs(run_dir(side, mode, k, seed)).items()}
    return cache[key]


def l1(a, b):
    out = []
    for f in a:
        n = min(a[f].shape[-1], b[f].shape[-1])
        assert abs(a[f].shape[-1] - b[f].shape[-1]) <= 1, (f, a[f].shape, b[f].shape)
        out.append(float((a[f][:, :n] - b[f][:, :n]).abs().mean()))
    return out


def cell(values):
    return f"{np.mean(values):.3f} (max {np.max(values):.3f})"


print("Whole-utterance log-mel L1, mean (max) over six utterances x five draws.\n")
print("| Euler steps | TT Stage 1 vs its 10 steps | reference Stage 1 vs its 10 steps | TT streaming vs its 10 steps | "
      "reference streaming vs its 10 steps | TT streaming vs upstream's, same steps |")  # fmt: skip
print("|---|---|---|---|---|---|")
for k in STEPS:
    cols = []
    for side, mode in (("tt", "stage1"), ("ref", "stage1"), ("tt", "stream"), ("ref", "stream")):
        cols.append("—" if k == 10 else cell([v for s in SEEDS for v in l1(logmel(side, mode, k, s), logmel(side, mode, 10, s))]))
    cols.append(cell([v for s in SEEDS for v in l1(logmel("tt", "stream", k, s), logmel("ref", "stream", k, s))]))
    print(f"| {k} | " + " | ".join(cols) + " |")

print("\nFor scale, two noise draws of the same side at 10 steps (seed 1 against seeds 2-5), mean (max):\n")
print("| TT Stage 1 | reference Stage 1 | TT streaming | reference streaming |")
print("|---|---|---|---|")
print("| " + " | ".join(cell([v for s in SEEDS[1:] for v in l1(logmel(side, mode, 10, 1), logmel(side, mode, 10, s))])
                        for side, mode in (("tt", "stage1"), ("ref", "stage1"), ("tt", "stream"), ("ref", "stream"))) + " |")
