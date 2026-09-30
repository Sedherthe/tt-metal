"""Why did the prototype's traced CFM (traced_cfm_prototype.py) produce the wrong audio, when the pipeline's own traced
solve (`_capture`, tested) does not? One bucket (the first chunk's, 512 frames), the conditioning of a real chunk,
the 10-step solve against the eager one:
- A: `TtCausalConditionalCFM`'s own traced path (`forward(use_trace=True)`: `_capture`, the output allocated inside
  the capture), its capture solve and a reuse;
- B: the prototype's slot, the body writing its result with `ttnn.add(..., output_tensor=next_x)` into a buffer
  allocated before the capture;
- C: the same slot, the body updating `x` in place (`ttnn.add_(x, step)`), so there is no output buffer and no copy
  after the replay.
Each is reported as PCC and max |diff| against the eager solve.

    python traced_cfm_diag2.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfm_step_profile as prof  # noqa: E402
import traced_cfm_prototype as proto  # noqa: E402

T = 512


def compare(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"pcc": round(float(np.corrcoef(a.numpy(), b.numpy())[0, 1]), 6), "max_abs_diff": round(float((a - b).abs().max()), 4)}


def body_inplace(cfm, slot, t_len):
    x2 = ttnn.concat([slot.x_buf, slot.x_buf], dim=0)
    d = cfm.estimator._forward_from_raw_temb(x2, slot.mask2_buf, slot.mu2_buf, slot.temb_raw_buf, slot.spks2_buf,
                                             slot.cond2_buf, t_len, batch_size=2, chunk_bias=slot.chunk_bias_buf)  # fmt: skip
    ttnn.deallocate(x2)
    c = ttnn.slice(d, [0, 0, 0], [1, t_len, proto.CH])
    u = ttnn.slice(d, [1, 0, 0], [2, t_len, proto.CH])
    ttnn.deallocate(d)
    guided = ttnn.subtract(ttnn.multiply(c, 1.0 + cfm.inference_cfg_rate), ttnn.multiply(u, cfm.inference_cfg_rate))
    step = ttnn.multiply(guided, slot.dt_buf)
    ttnn.add_(slot.x_buf, step)


def solve_with_slot(cfm, slot, mu, mask, spks, c, inplace: bool):
    """`_forward_traced`'s reuse path, by hand: refill the slot, replay per step, read x."""
    dev = prof.upload_conditioning(cfm, mu, mask, spks, c)
    z = ttnn.from_torch(cfm.rand_noise[:, :T, :].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=cfm.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    for src, dst in zip((*dev, z), (slot.mu2_buf, slot.spks2_buf, slot.cond2_buf, slot.mask2_buf, slot.x_buf)):
        ttnn.copy(src, dst)
        ttnn.deallocate(src)
    ttnn.synchronize_device(cfm.device)
    for t_val, dt_val in cfm._euler_schedule(prof.t_span_for(cfm, 10, mu.dtype)):
        temb, dt = cfm._temb_device(t_val), cfm._dt_device(dt_val)
        ttnn.copy(temb, slot.temb_raw_buf)
        ttnn.copy(dt, slot.dt_buf)
        ttnn.deallocate(temb)
        ttnn.deallocate(dt)
        ttnn.execute_trace(cfm.device, slot.trace_id, cq_id=0, blocking=True)
        if not inplace:
            ttnn.copy(slot.next_x, slot.x_buf)
    return ttnn.to_torch(slot.x_buf).float().reshape(1, T, proto.CH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=200_000_000)
    out = {}
    try:
        flow = prof.build_flow(device)
        cfm = flow.decoder
        g = prof.capture_conditioning(flow, {"stream_256": ("zero_shot_121-127105-0015", 32 + prof.PRE_LOOKAHEAD, True)})
        mu, mask, spks, c = g["stream_256"]["args"]
        ref = cfm.forward(mu, mask, 10, spks, c, use_trace=False, streaming=True)
        a1 = cfm.forward(mu, mask, 10, spks, c, use_trace=True, streaming=True)
        a2 = cfm.forward(mu, mask, 10, spks, c, use_trace=True, streaming=True)
        cfm.release_cfm_trace()
        out["A_capture_solve"], out["A_reuse"] = compare(a1, ref), compare(a2, ref)
        for name, inplace in (("B_output_tensor", False), ("C_inplace_add", True)):
            slot = proto.make_slot(cfm, T)
            if inplace:
                ttnn.deallocate(slot.next_x)
                slot.next_x = None
                orig = proto.body
                proto.body = lambda cfm_, slot_, t_: body_inplace(cfm_, slot_, t_)
            try:
                proto.compile_slot(cfm, slot, T) if not inplace else [body_inplace(cfm, slot, T) for _ in range(2)]
                proto.capture_slot(cfm, slot, T)
                r1 = solve_with_slot(cfm, slot, mu, mask, spks, c, inplace)
                r2 = solve_with_slot(cfm, slot, mu, mask, spks, c, inplace)
                out[name] = {"first": compare(r1, ref), "second": compare(r2, ref)}
            except Exception as e:  # noqa: BLE001
                out[name] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
            finally:
                if inplace:
                    proto.body = orig
                if slot.trace_id is not None:
                    ttnn.release_trace(device, slot.trace_id)
            print(json.dumps({name: out[name]}), flush=True)
        print(json.dumps({k: out[k] for k in ("A_capture_solve", "A_reuse")}), flush=True)
    finally:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
