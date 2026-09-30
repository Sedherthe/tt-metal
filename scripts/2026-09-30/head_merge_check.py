"""Stage 3, lever (a): the CFM's attention heads merged by `ttnn.experimental.nlp_concat_heads` against the committed
transpose + reshape, on the same real inputs, in one process.

`TtBasicTransformerBlock.__call__` as committed (`--before`, default `33aa3601eb`) is compiled verbatim from git in
the current module's namespace. The working tree's is the new one. For each geometry `cfm_step_profile.py` captures
from real chunks (first chunk 512 frames, a 100-token hop 768, a later chunk 1,024, a non-streaming final 640), both
versions run:
- one estimator call at the first Euler step (`_forward_from_raw_temb`, batch 2, CFG);
- the whole 10-step eager solve.

The result is max |diff| and PCC between the two, per geometry, and whether they are bit-identical.

    python head_merge_check.py --out <json> [--before 33aa3601eb]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfm_step_profile as prof  # noqa: E402

from models.experimental.cosyvoice2.tt.flow import decoder as dec  # noqa: E402


def committed_block_call(rev: str):
    src = subprocess.check_output(
        ["git", "-C", "/home/user/tt-metal", "show", f"{rev}:models/experimental/cosyvoice2/tt/flow/decoder.py"], text=True
    )
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "TtBasicTransformerBlock")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    body = textwrap.dedent(ast.get_source_segment(src, fn, padded=True))
    assert "nlp_concat_heads" not in body and "ttnn.transpose(out, 1, 2)" in body, "not the committed merge"
    namespace = dict(vars(dec))
    exec(compile(body, f"{rev}:tt/flow/decoder.py", "exec"), namespace)  # noqa: S102
    return namespace["__call__"]


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    return {"max_abs_diff": float((a - b).abs().max()), "pcc": float(np.corrcoef(a.numpy(), b.numpy())[0, 1]),
            "identical": bool(torch.equal(a, b))}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", default="33aa3601eb")
    args = ap.parse_args()
    import inspect

    new_call, old_call = dec.TtBasicTransformerBlock.__call__, committed_block_call(args.before)
    assert "nlp_concat_heads" in inspect.getsource(new_call), "the working tree's block does not merge with nlp_concat_heads"
    geometries = {
        "stream_256": ("zero_shot_121-127105-0015", 32 + prof.PRE_LOOKAHEAD, True),
        "stream_384": ("zero_shot_121-127105-0024", 182 + prof.PRE_LOOKAHEAD, True),
        "stream_512": ("zero_shot_121-127105-0024", 282 + prof.PRE_LOOKAHEAD, True),
        "final_320": ("zero_shot_121-127105-0015", 95, False),
    }
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)
    report = {"before": args.before, "geometries": {}}
    try:
        flow = prof.build_flow(device)
        cfm = flow.decoder
        cond = prof.capture_conditioning(flow, geometries)  # compiles the new merge's programs
        for name, g in cond.items():
            mu, mask, spks, c = g["args"]
            t_len = mu.shape[1]
            padded = not g["streaming"] and not bool((mask != 0).all())
            out = {}
            for version, call in (("new", new_call), ("old", old_call)):
                dec.TtBasicTransformerBlock.__call__ = call
                dev = prof.upload_conditioning(cfm, mu, mask, spks, c)
                chunk_bias = prof.chunk_bias_for(cfm, t_len, g["streaming"], padded)
                t_span = prof.t_span_for(cfm, 10, mu.dtype)
                x = cfm.rand_noise[:, :t_len, :].to(mu.dtype)
                x_dev = ttnn.from_torch(torch.cat([x, x], 0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                temb = cfm.estimator._sinusoidal_pos_emb(t_span[0].unsqueeze(0).repeat(2))
                mu_dev, spks_dev, cond_dev, mask_dev = dev
                step = cfm.estimator._forward_from_raw_temb(x_dev, mask_dev, mu_dev, temb, spks_dev, cond_dev, t_len,
                                                            batch_size=2, chunk_bias=chunk_bias)  # fmt: skip
                out[version] = {"step": ttnn.to_torch(step).float()}
                for tensor in (*dev, x_dev, temb, step, *([chunk_bias] if chunk_bias is not None else [])):
                    ttnn.deallocate(tensor)
                out[version]["solve"] = cfm.forward(mu, mask, 10, spks, c, use_trace=False, streaming=g["streaming"])
            dec.TtBasicTransformerBlock.__call__ = new_call
            row = {"t_len": t_len, "streaming": g["streaming"],
                   "one_step_dphi": compare(out["new"]["step"], out["old"]["step"]),
                   "ten_step_solve": compare(out["new"]["solve"], out["old"]["solve"])}  # fmt: skip
            report["geometries"][name] = row
            print(json.dumps({name: row}), flush=True)
    finally:
        dec.TtBasicTransformerBlock.__call__ = new_call
        ttnn.close_device(device)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
