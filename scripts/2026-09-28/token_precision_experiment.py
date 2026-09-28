"""Token accuracy vs LLM precision. The same teacher-forced comparison as tests/e2e/test_token_accuracy.py, with the
Qwen2 decoder built (a) at tt_transformers' default (DecodersPrecision.accuracy for Qwen2: attention bf16 at HiFi4,
MLP weights bfp8 with fp16 accumulation), and (b) everything bf16 at HiFi4. Also decode speed for (b)."""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

import ttnn
from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file
from models.experimental.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config
from models.experimental.cosyvoice2.tt.prompt import PromptContext
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)

arm = sys.argv[1]  # "default" | "bf16" | "fp32head" | "fp32head_hifi3" | "fp32out"

if arm.startswith("fp32head") or arm == "fp32out":
    # Only the output head changes: fp32 weights, the bf16 hidden cast to fp32, HiFi4 with fp32 accumulation, fp32
    # logits. bf16 logits (~8 in magnitude) resolve steps of ~0.06, about the size of the near-ties TT flips.
    from models.common.lightweightmodule import LightweightModule
    from models.experimental.cosyvoice2.tt.llm import qwen2lm

    # fp32out: bf16 weights and input as today, but fp32 accumulation and fp32 logits out
    wdtype = ttnn.bfloat16 if arm == "fp32out" else ttnn.float32
    fidelity = ttnn.MathFidelity.HiFi3 if arm == "fp32head_hifi3" else ttnn.MathFidelity.HiFi4

    def _init(self, device, weight, bias, dtype=None):
        LightweightModule.__init__(self)
        self.device = device
        self.weight = ttnn.from_torch(weight.detach().float().t().contiguous(), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.bias = ttnn.from_torch(bias.detach().float().reshape(1, 1, -1), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fidelity,
                                                         math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)

    def _forward(self, x):
        if wdtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        return ttnn.linear(x, self.weight, bias=self.bias, compute_kernel_config=self.cc, dtype=ttnn.float32)

    qwen2lm.TtLinearHead.__init__, qwen2lm.TtLinearHead.forward = _init, _forward
os.environ["HF_MODEL"] = os.path.expanduser("~/.cache/cosyvoice2_ttnn/qwen2_backbone")
refs = sorted(glob.glob("/home/user/data/cosyvoice2_token_accuracy/*.npz"))
device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=100_000_000)
try:
    opt = None
    if arm == "bf16":
        settings = {
            "TensorPrecision": {g: PrecisionSetting.BF16 for g in (TensorGroup.FF1_FF3, TensorGroup.FF2, TensorGroup.WQKV, TensorGroup.WO, TensorGroup.KV_CACHE)},
            "OpFidelity": {g: MathFidelitySetting.HIFI4 for g in OpGroup if g != OpGroup.ACCURACY},
        }
        opt = lambda a: DecodersPrecision(a.n_layers, a.model_name, ModelOptimizations(settings))  # noqa: E731
    args = ModelArgs(device, max_batch_size=1, max_seq_len=CosyVoice2Config().max_seq_len(), dummy_weights=False,
                     use_hf_rope=True, optimizations=opt)
    llm = TtQwen2LM(args, device, args.load_state_dict(), cosyvoice_state_dict=load_checkpoint_file("llm.pt"), use_decode_trace=True)
    agree = total = 0
    steps_s, steps = 0.0, 0
    for path in refs:
        case_id = os.path.basename(path)[:-4]
        ref = np.load(path)
        ctx = PromptContext.from_npz(f"/home/user/data/cosyvoice2_inputs/{case_id}.npz")
        text = torch.cat([ctx.prompt_text_ids.long(), torch.tensor([ctx.meta["segment_text_ids"][0]])], dim=1)
        t0 = time.perf_counter()
        idx, _ = llm.teacher_forced_topk(text, ctx.llm_prompt_speech_tokens.long(), ref["tokens"].tolist())
        steps_s += time.perf_counter() - t0
        steps += len(ref["tokens"])
        a = (idx[:, 0].numpy() == ref["top5"][:, 0])
        agree += int(a.sum()); total += len(a)
        print(f"  {arm} {case_id:<34} {100 * a.mean():6.2f} %", flush=True)
    print(json.dumps({"arm": arm, "top1_agreement_percent": round(100 * agree / total, 2), "positions": total,
                      "seconds_per_token_incl_prefill": round(steps_s / steps, 4)}))
finally:
    ttnn.close_device(device)
