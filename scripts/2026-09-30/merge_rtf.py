"""Lever (a), the RTF it gives: the five noise draws of `noise_draws.py` before the head merge (D43's, 2026-09-30
06:03, the committed merge) and after it (`cosyvoice2_merge`). The protocol is identical: one process, both warm-ups,
the six utterances per draw, Stage 1 then streaming, nothing else on the host. `noise_draws.py` records each
utterance's RTF and audio duration. Wall is RTF times audio, and the aggregate is the sum of wall over the sum of audio.

    python3 merge_rtf.py > merge_rtf.md
"""
import json

import numpy as np

RUNS = {"before (D43, transpose + reshape)": "/home/user/data/cosyvoice2_draws",
        "after (nlp_concat_heads)": "/home/user/data/cosyvoice2_merge"}  # fmt: skip
SEEDS = (1, 2, 3, 4, 5)

for mode in ("stage1", "stream"):
    print(f"\n## {'Stage 1' if mode == 'stage1' else 'Streaming'} RTF over five draws\n")
    print("| | worst per draw | aggregate per draw | per utterance, mean over the draws |")
    print("|---|---|---|---|")
    per_case = {}
    for name, d in RUNS.items():
        worst, agg = [], []
        for s in SEEDS:
            rows = json.load(open(f"{d}/tt_{mode}_seed{s}/results.json"))["results"]
            worst.append(max(r["rtf"] for r in rows))
            agg.append(sum(r["rtf"] * r["audio_s"] for r in rows) / sum(r["audio_s"] for r in rows))
            for r in rows:
                per_case.setdefault(r["case_id"], {}).setdefault(name, []).append(r["rtf"])
        cases = " ".join(f"{c[len('zero_shot_'):]}: {np.mean(v[name]):.3f}" for c, v in sorted(per_case.items()))
        print(f"| {name} | {min(worst):.3f}–{max(worst):.3f} | {min(agg):.3f}–{max(agg):.3f} | {cases} |")
