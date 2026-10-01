"""Is this card bitwise reproducible? The same device op, repeated on the same inputs, compared bit for bit.

Deterministic kernels give identical bits on every repeat. B44: on the 10-01 card, the same LLM request in one
process sampled different tokens on its second call, which the old cards never did.

Each case runs `--repeats` times on one input and compares every output with the first (torch.equal). It prints the
number of mismatching repeats, the mismatching elements and the largest difference. The cases:
- matmul, bf16, the LLM's attention matmuls' config (HiFi4, fp32 accumulation), multi-core;
- matmul, bfp8 weights (the LLM MLP's), HiFi2;
- softmax and rms_norm over a decode-sized row block.
"""
import argparse
import json
import time

import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=200)
ap.add_argument("--out", required=True)
args = ap.parse_args()

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)


def to_dev(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


a = to_dev(torch.randn(1, 1, 1024, 896))
b16 = to_dev(torch.randn(1, 1, 896, 4864) / 30)
b8 = to_dev(torch.randn(1, 1, 896, 4864) / 30, dtype=ttnn.bfloat8_b)
x = to_dev(torch.randn(1, 1, 32, 6592) * 4)
# the norm's gamma: ROW_MAJOR, its last dim the tile width (layernorm_device_operation.cpp:106)
w = ttnn.from_torch(
    (torch.rand(6592) + 0.5).reshape(1, 1, 6592 // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
)
cases = {
    "matmul_bf16_hifi4": lambda: ttnn.matmul(a, b16, compute_kernel_config=hifi4),
    "matmul_bfp8_hifi2": lambda: ttnn.matmul(a, b8, compute_kernel_config=hifi2),
    "softmax": lambda: ttnn.softmax(x, dim=-1),
    "rms_norm": lambda: ttnn.rms_norm(x, weight=w, epsilon=1e-6),
}
results = {}
for name, fn in cases.items():
    t0 = time.time()
    first = ttnn.to_torch(fn())
    bad = elems = 0
    worst = 0.0
    for _ in range(args.repeats - 1):
        out = ttnn.to_torch(fn())
        if not torch.equal(out, first):
            bad += 1
            d = (out.float() - first.float()).abs()
            elems += int((d > 0).sum())
            worst = max(worst, float(d.max()))
    results[name] = {"repeats": args.repeats, "mismatching_repeats": bad, "mismatching_elements": elems,
                     "max_abs_diff": worst, "seconds": round(time.time() - t0, 1)}
    print(name, json.dumps(results[name]), flush=True)
ttnn.close_device(device)
json.dump(results, open(args.out, "w"), indent=1)
print("DETERMINISTIC" if all(r["mismatching_repeats"] == 0 for r in results.values()) else "NOT DETERMINISTIC")
