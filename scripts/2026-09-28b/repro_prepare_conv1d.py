"""Standalone reproducer: ttnn.conv1d with prepare_conv_weights/prepare_conv_bias vs the raw weight, at the three
CosyVoice2 HiFT conv geometries where the prepared path is wrong (N150). Random weights and inputs (seeded), fp32
activations, bf16 weights, HiFi4 with fp32 accumulation (CosyVoice2's HiFT config). Each case runs with the input in
DRAM (conv1d auto-slices it through DRAM) and in L1 (conv1d runs fully in L1, no slicing), and with prepare given
the conv's compute config or not. Relative L2 error against a float64 torch conv."""
import json

import torch

import ttnn

CASES = [  # name, in, out, kernel, stride, padding, dilation, input length
    ("resblock k=11", 128, 128, 11, 1, 5, 1, 5120),
    ("resblock k=11 d=3", 128, 128, 11, 1, 15, 3, 5120),
    ("source_downs[0] k=30 s=15", 18, 256, 30, 15, 7, 1, 76801),
    ("source_downs[1] k=6 s=3", 18, 128, 6, 3, 1, 1, 107521),
    # neighbours that are fine in the pipeline, for contrast
    ("resblock k=11, L=10240", 128, 128, 11, 1, 5, 1, 10240),
    ("source_downs[0], L=61441", 18, 256, 30, 15, 7, 1, 61441),
]

device = ttnn.open_device(device_id=0, l1_small_size=65536)
cc = ttnn.init_device_compute_kernel_config(
    device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def cfg(**kw):
    return ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, deallocate_activation=False, config_tensors_in_dram=True, **kw)


def slices(n):
    return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth, num_slices=n)


# variant name -> (input memory, conv config, slice config passed to BOTH prepare and conv (None: not passed))
VARIANTS = {
    "DRAM input, auto slicing": (ttnn.DRAM_MEMORY_CONFIG, cfg(), None),
    "DRAM input, 2 slices": (ttnn.DRAM_MEMORY_CONFIG, cfg(), slices(2)),
    "DRAM input, 8 slices": (ttnn.DRAM_MEMORY_CONFIG, cfg(), slices(8)),
    "DRAM input, act_block_h_override=1024": (ttnn.DRAM_MEMORY_CONFIG, cfg(act_block_h_override=1024), None),
    "L1 input (no slicing)": (ttnn.L1_MEMORY_CONFIG, cfg(), None),
}
rows = []
try:
    for name, cin, cout, k, s, pad, dil, L in CASES:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(cout, cin, k, generator=g) / (cin * k) ** 0.5
        b = torch.randn(cout, generator=g) * 0.1
        x = torch.randn(1, L, cin, generator=g)
        ref = torch.nn.functional.conv1d(x.transpose(1, 2).double(), w.double(), b.double(), stride=s, padding=pad,
                                         dilation=dil).transpose(1, 2)[0]
        w4 = ttnn.from_torch(w.unsqueeze(2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        w3 = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        b4 = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        for variant, (mem, conv_config, slice_config) in VARIANTS.items():
            res = {"case": name, "L": L, "variant": variant}
            try:
                xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
            except Exception as e:  # noqa: BLE001
                res["error"] = f"input: {type(e).__name__}: {str(e)[:100]}"
                print(json.dumps(res), flush=True)
                continue
            extra = {} if slice_config is None else {"slice_config": slice_config}

            def conv(weight, bias):
                try:
                    out, n = ttnn.conv1d(input_tensor=xd, weight_tensor=weight, bias_tensor=bias, device=device,
                                         in_channels=cin, out_channels=cout, batch_size=1, input_length=L,
                                         kernel_size=k, stride=s, padding=pad, dilation=dil, groups=1,
                                         conv_config=conv_config, compute_config=cc, dtype=ttnn.float32,
                                         return_output_dim=True, **extra)
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    i = msg.find("info:")
                    return f"{type(e).__name__}: {msg[i + 5:i + 130].strip() if i >= 0 else msg[:120]}"
                got = ttnn.to_torch(out).double().reshape(n, cout)
                ttnn.deallocate(out)
                return round(float((got - ref).norm() / ref.norm()), 6)

            kw = dict(input_memory_config=xd.memory_config(), input_layout=xd.layout, in_channels=cin,
                      out_channels=cout, batch_size=1, input_height=1, input_width=L, kernel_size=(1, k),
                      stride=(1, s), padding=(0, pad), dilation=(1, dil), groups=1, device=device,
                      input_dtype=ttnn.float32, conv_config=conv_config, **extra)
            for arm, arm_extra in (("prepared", {}), ("prepared, compute config", {"compute_config": cc})):
                try:
                    pw = ttnn.prepare_conv_weights(weight_tensor=w4, weights_format="OIHW", has_bias=True, **kw,
                                                   **arm_extra)
                    pb = ttnn.prepare_conv_bias(bias_tensor=b4, **kw, **arm_extra)
                    res[arm] = conv(pw, pb)
                except Exception as e:  # noqa: BLE001
                    res[arm] = f"{type(e).__name__}: {str(e)[:100]}"
            res["raw"] = conv(w3, b4)
            ttnn.deallocate(xd)
            rows.append(res)
            print(json.dumps(res), flush=True)
finally:
    ttnn.close_device(device)
