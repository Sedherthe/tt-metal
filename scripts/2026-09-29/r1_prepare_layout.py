"""R1: prepare_conv_weights declared TILE (what TtConv1d passes: the activation's layout) vs declared ROW_MAJOR vs the
raw weight, relative L2 error against a float64 torch conv, at the rebuild spec's geometries plus the streaming HiFT
lengths (108, 128, 208 frames) R3/R4 will run. The pipeline's settings: input TILE in DRAM (conv1d auto-slices it),
bf16 weights, HiFi4 + fp32 accumulation + packer L1 accumulation, config tensors in DRAM. HiFT convs run fp32
activations, the flow's causal convs bf16 at batch 2 (CFG). Random weights and inputs, seeded per case.

    /opt/venv/bin/python r1_prepare_layout.py > r1_prepare_layout.log

One JSON line per case, then a Markdown table.
"""
import json

import torch

import ttnn


def hift_len(frames):  # source_downs input: the NSF source's STFT frames, 120 per mel frame, plus one
    return 120 * frames + 1


CASES = []  # name, in, out, kernel, stride, padding, dilation, length, batch, dtype
for frames, L in ((128, 5120), (256, 10240), (108, 4320), (208, 8320)):  # HiFT stage 1: 40 samples per mel frame
    for d, pad in ((1, 5), (3, 15), (5, 25)):
        if frames == 256 and d != 1:
            continue
        CASES.append((f"resblock k=11 d={d}, {frames} frames", 128, 128, 11, 1, pad, d, L, 1, "fp32"))
for frames in (640, 768, 896, 1024, 1280, 1536, 1792, 2048, 512, 256, 208, 128, 108):
    CASES.append((f"source_downs[0] k=30 s=15, {frames} frames", 18, 256, 30, 15, 7, 1, hift_len(frames), 1, "fp32"))
for frames in (896, 1024, 1280, 1536, 1792, 2048, 512, 256, 208, 128, 108):
    CASES.append((f"source_downs[1] k=6 s=3, {frames} frames", 18, 128, 6, 3, 1, 1, hift_len(frames), 1, "fp32"))
CASES.append(("flow CFM conv 320->256 k=3, 2560-token bucket", 320, 256, 3, 1, (2, 0), 1, 5120, 2, "bf16"))
CASES.append(("flow CFM conv 256->256 k=3, 2560-token bucket", 256, 256, 3, 1, (2, 0), 1, 5120, 2, "bf16"))

DTYPES = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}
device = ttnn.open_device(device_id=0, l1_small_size=65536)
cc = ttnn.init_device_compute_kernel_config(
    device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, deallocate_activation=False, config_tensors_in_dram=True)
rows = []
try:
    for name, cin, cout, k, s, pad, dil, L, batch, dt in CASES:
        dtype = DTYPES[dt]
        g = torch.Generator().manual_seed(0)
        w = torch.randn(cout, cin, k, generator=g) / (cin * k) ** 0.5
        b = torch.randn(cout, generator=g) * 0.1
        x = torch.randn(batch, L, cin, generator=g)
        pl, pr = (pad, pad) if isinstance(pad, int) else pad
        xh = torch.nn.functional.pad(x.transpose(1, 2).double(), (pl, pr))
        ref = torch.nn.functional.conv1d(xh, w.double(), b.double(), stride=s, dilation=dil).transpose(1, 2)
        n_out = ref.shape[1]
        w4 = ttnn.from_torch(w.unsqueeze(2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        w3 = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        b4 = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        xd = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def conv(weight, bias):
            try:
                out, n = ttnn.conv1d(
                    input_tensor=xd, weight_tensor=weight, bias_tensor=bias, device=device, in_channels=cin,
                    out_channels=cout, batch_size=batch, input_length=L, kernel_size=k, stride=s, padding=pad,
                    dilation=dil, groups=1, conv_config=conv_config, compute_config=cc, dtype=dtype,
                    return_output_dim=True,
                )
            except Exception as e:  # noqa: BLE001
                return f"{type(e).__name__}: {str(e)[:100]}"
            got = ttnn.to_torch(out).double().reshape(batch, n, cout)
            ttnn.deallocate(out)
            assert n == n_out, (n, n_out)
            err = float((got - ref).norm() / ref.norm())
            return err if err == err else float("inf")

        res = {"case": name, "L": L, "batch": batch, "dtype": dt}
        pad2d = (0, pad) if isinstance(pad, int) else (0, 0, *pad)
        for arm, layout in (("declared TILE", ttnn.TILE_LAYOUT), ("declared ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT)):
            kw = dict(
                input_memory_config=xd.memory_config(), input_layout=layout, in_channels=cin, out_channels=cout,
                batch_size=batch, input_height=1, input_width=L, kernel_size=(1, k), stride=(1, s), padding=pad2d,
                dilation=(1, dil), groups=1, device=device, input_dtype=dtype, conv_config=conv_config,
            )
            try:
                pw = ttnn.prepare_conv_weights(weight_tensor=w4, weights_format="OIHW", has_bias=True, **kw)
                pb = ttnn.prepare_conv_bias(bias_tensor=b4, **kw)
                res[arm] = conv(pw, pb)
                ttnn.deallocate(pw)
                ttnn.deallocate(pb)
            except Exception as e:  # noqa: BLE001
                res[arm] = f"{type(e).__name__}: {str(e)[:100]}"
        res["raw weight"] = conv(w3, b4)
        ttnn.deallocate(xd)
        rows.append(res)
        print(json.dumps(res), flush=True)
finally:
    ttnn.close_device(device)


def fmt(v):
    return v if isinstance(v, str) else ("inf" if v == float("inf") else f"{v:.3g}")


print("\n| case | length | declared TILE | declared ROW_MAJOR | raw weight |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r['case']} | {r['L']:,} | {fmt(r['declared TILE'])} | {fmt(r['declared ROW_MAJOR'])} | {fmt(r['raw weight'])} |")
