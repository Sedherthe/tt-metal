"""B44: is the LLM itself reproducible on this card? One forced sequence, repeated in one process.

`TtQwen2LM.teacher_forced_topk` (the token-accuracy test's path: the prefill, then one decode step per forced token)
runs `--repeats` times on the same case, traced and eager. Every repeat's top-5 values are compared with the first's,
bit for bit. Row 0 is the prefill's last position; row i+1 the decode step after forced token i. So the first
differing row says whether the prefill or the decode differs, and from which step.
"""
import argparse
import json
import os

import numpy as np
import torch
import ttnn

from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN
from models.experimental.cosyvoice2.tt.prompt import PromptContext

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="zero_shot_121-127105-0015")
ap.add_argument("--repeats", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()

inputs, token_ref = os.environ["COSYVOICE2_INPUTS"], os.environ["COSYVOICE2_TOKEN_REF"]
device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)
pipe = CosyVoice2TTNN(device)
ctx = PromptContext.from_npz(os.path.join(inputs, f"{args.case}.npz"))
text_ids = torch.cat(
    [ctx.prompt_text_ids.long(), torch.tensor([ctx.meta["segment_text_ids"][0]], dtype=torch.long)], dim=1
)
forced = np.load(os.path.join(token_ref, f"{args.case}.npz"))["tokens"].tolist()
out = {"case": args.case, "positions": len(forced) + 1}
for use_trace in (True, False):
    runs = []
    for r in range(args.repeats):
        idx, val = pipe.llm.teacher_forced_topk(
            text_ids, ctx.llm_prompt_speech_tokens.long(), forced, k=5, use_trace=use_trace
        )
        runs.append((idx.clone(), val.clone()))
    rows = []
    for r, (idx, val) in enumerate(runs[1:], start=1):
        same = (val == runs[0][1]).all(dim=1) & (idx == runs[0][0]).all(dim=1)
        bad = (~same).nonzero().flatten().tolist()
        rows.append({
            "repeat": r, "differing_rows": len(bad), "first_differing_row": bad[0] if bad else None,
            "max_abs_diff": float((val.float() - runs[0][1].float()).abs().max()),
            "top1_flips": int((idx[:, 0] != runs[0][0][:, 0]).sum()),
        })
        print(("traced" if use_trace else "eager"), json.dumps(rows[-1]), flush=True)
    out["traced" if use_trace else "eager"] = rows
ttnn.close_device(device)
json.dump(out, open(args.out, "w"), indent=1)
ok = all(r["differing_rows"] == 0 for k in ("traced", "eager") for r in out[k])
print("LLM REPRODUCIBLE" if ok else "LLM NOT REPRODUCIBLE")
