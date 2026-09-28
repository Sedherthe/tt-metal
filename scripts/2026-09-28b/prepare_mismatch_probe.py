"""Are the 20 corrupted HiFT conv geometries (B15) prepare_conv_weights being called with a different compute config
than the conv itself? TtConv1d._prepared passes neither compute_config nor slice_config; the conv runs HiFi4 with
fp32 accumulation. Inside real HiFT runs at the three buckets where the corruption showed (128, 640, 896 frames),
every conv's first-sight check is intercepted and, on the actual input tensor, four weight paths are compared with a
float64 host conv:
  prep_as_pipeline   prepare_conv_weights(conv_config) -- what TtConv1d does today
  prep_with_cc       prepare_conv_weights(conv_config, compute_config=<the conv's own>)
  prep_with_cc_slice prepare_conv_weights(conv_config, compute_config, slice_config=DRAM width auto), conv the same
  raw                the unprepared weight (the op prepares it internally)
"""
import json
import sys

import torch

import ttnn
from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.experimental.cosyvoice2.tt.hifigan import conv as conv_mod
from models.experimental.cosyvoice2.tt.hifigan.conv import config_tensors_in_dram_override, relative_error
from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.experimental.cosyvoice2.tt.hifigan.generator import (
    TorchHiFTDecodeRef,
    TorchHiFTGeneratorInferenceRef,
    TtHiFTDecoder,
    TtHiFTGenerator,
)

FRAMES = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [128, 640, 896]
rows = []
orig = conv_mod.TtConv1d._verify_and_resolve


def probe(self, x, weight, bias, input_length, batch_size, out, key):
    pad2d = (0, self.padding) if isinstance(self.padding, int) else (0, 0, *self.padding)
    kw = dict(
        input_memory_config=x.memory_config(), input_layout=x.layout, in_channels=self.in_channels,
        out_channels=self.out_channels, batch_size=batch_size, input_height=1, input_width=input_length,
        kernel_size=(1, self.kernel_size), stride=(1, self.stride), padding=pad2d, dilation=(1, self.dilation),
        groups=self.groups, device=self.device, input_dtype=self.dtype, conv_config=self.conv_config,
    )
    ref = self._host_reference(x, input_length, batch_size)
    out_len = ref.shape[1]
    res = {"conv": f"Conv1d({self.in_channels}->{self.out_channels}, k={self.kernel_size}, s={self.stride}, "
                   f"d={self.dilation})", "L": input_length,
           "input": f"{x.memory_config().buffer_type}/{x.memory_config().memory_layout}/{x.layout}/{x.dtype}"}

    def err(t):
        h = self._to_host(t, batch_size, out_len)
        return float(relative_error(h, ref)), bool(torch.isinf(h).any() or torch.isnan(h).any())

    res["prep_as_pipeline"] = err(out)
    for name, extra, conv_extra in (
        ("prep_with_cc", {"compute_config": self.compute_config}, {}),
        ("prep_with_cc_slice", {"compute_config": self.compute_config,
                                "slice_config": ttnn.Conv2dSliceConfig(
                                    slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth, num_slices=0)},
         {"slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth,
                                                 num_slices=0)}),
    ):
        try:
            w = ttnn.prepare_conv_weights(weight_tensor=self._weight_4d, weights_format="OIHW",
                                          has_bias=self.bias is not None, **kw, **extra)
            b = ttnn.prepare_conv_bias(bias_tensor=self.bias, **kw, **extra) if self.bias is not None else None
            t, _ = ttnn.conv1d(input_tensor=x, weight_tensor=w, bias_tensor=b, device=self.device,
                               in_channels=self.in_channels, out_channels=self.out_channels, batch_size=batch_size,
                               input_length=input_length, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, groups=self.groups,
                               conv_config=self.conv_config, compute_config=self.compute_config, dtype=self.dtype,
                               return_output_dim=True, **conv_extra)
            res[name] = err(t)
            ttnn.deallocate(t)
        except Exception as e:  # noqa: BLE001
            res[name] = f"{type(e).__name__}: {str(e)[:160]}"
    raw, _ = self._conv(x, self.weight, self.bias, input_length, batch_size, self.compute_config)
    res["raw"] = err(raw)
    ttnn.deallocate(raw)
    rows.append(res)
    bad = res["prep_as_pipeline"][0] > 0.05
    if bad or any(isinstance(res[k], tuple) and res[k][0] > 0.05 for k in ("prep_with_cc", "prep_with_cc_slice")):
        print(json.dumps(res), flush=True)
    return orig(self, x, weight, bias, input_length, batch_size, out, key)


conv_mod.TtConv1d._verify_and_resolve = probe

device = ttnn.open_device(device_id=0, l1_small_size=65536)
try:
    with config_tensors_in_dram_override(True):
        hift_sd = load_checkpoint_file("hift.pt")
        decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
        ref = TorchHiFTGeneratorInferenceRef(
            decode_ref, TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor.")),
            hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"])
        gen = TtHiFTGenerator(device, ref, TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32), dtype=ttnn.float32)
    g = torch.Generator().manual_seed(0)
    for frames in FRAMES:
        mel = (torch.randn(1, frames, 80, generator=g) * 2.0 - 6.0).clamp(min=-11.5)
        mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        noise = torch.randn(1, frames * gen.upsample_scale, 9, generator=g)
        ttnn.deallocate(gen.inference(mel_dev, frames, 1, sine_noise=noise))
        ttnn.deallocate(mel_dev)
        print(f"--- {frames} frames done: {len(rows)} convs probed so far", flush=True)
finally:
    ttnn.close_device(device)

summary = {}
for r in rows:
    for k in ("prep_as_pipeline", "prep_with_cc", "prep_with_cc_slice", "raw"):
        v = r[k]
        if isinstance(v, tuple):
            summary.setdefault(k, {"n": 0, "bad": 0, "max": 0.0})
            summary[k]["n"] += 1
            summary[k]["bad"] += v[0] > 0.05
            summary[k]["max"] = max(summary[k]["max"], v[0])
print(json.dumps({"probed": len(rows), "summary": summary}))
with open("/home/user/data/cosyvoice2_runs/0928b/prepare_mismatch_probe.json", "w") as fh:
    json.dump(rows, fh, indent=1)
