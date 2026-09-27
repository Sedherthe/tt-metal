"""(f) Does the fp32 F0/source default change RTF? One process, one pipeline (reported config: F0/source fp32), plus a
second TtHiFTGenerator with bf16 F0/source sharing the SAME fp32 TtHiFTDecoder. Per corpus case: LLM + flow once
(seed 1986), then HiFT on that one mel, alternating fp32/bf16, REPS times each. The first call of each is first
sight; the rest are warm. Reports warm HiFT medians and the RTF difference they imply."""
import glob
import os
import statistics
import sys
import time

import torch

import ttnn
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
    TorchHiFTDecodeRef,
    TorchHiFTGeneratorInferenceRef,
    TtHiFTGenerator,
)
from models.demos.audio.cosyvoice2.tt.pipeline import CosyVoice2TTNN
from models.demos.audio.cosyvoice2.tt.prompt import PromptContext, RandomSources

REPS = 4
cases = sys.argv[1].split(",") if len(sys.argv) > 1 else None
ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(os.environ["COSYVOICE2_INPUTS"], "*.npz")))]
ctxs = [c for c in ctxs if c.meta["case"]["set"] == "librispeech" and (not cases or c.meta["case"]["case_id"] in cases)]
device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)
rows = []
try:
    pipe = CosyVoice2TTNN(device)
    hift_sd = load_checkpoint_file("hift.pt")
    ref = TorchHiFTGeneratorInferenceRef(
        TorchHiFTDecodeRef.from_checkpoint(hift_sd),
        TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor.")),
        hift_sd["m_source.l_linear.weight"],
        hift_sd["m_source.l_linear.bias"],
    )
    gens = {"float32": pipe.hift, "bfloat16": TtHiFTGenerator(device, ref, pipe.hift.decoder, dtype=ttnn.bfloat16)}
    for ctx in ctxs:
        cid = ctx.meta["case"]["case_id"]
        ids = pipe.text.encode(ctx.meta["segments"][0])
        tokens = pipe.text_to_tokens(ctx, ids, seed=1986)
        mel = pipe.tokens_to_mel(tokens, ctx)
        frames = int(mel.shape[1])
        audio_s = frames * 480 / 24000
        noise = torch.randn(1, frames * 480, 9)
        times = {d: [] for d in gens}
        wavs = {}
        for rep in range(REPS):
            for d, gen in gens.items():
                pipe.hift = gen
                mel_dev = ttnn.from_torch(mel, dtype=getattr(ttnn, d), layout=ttnn.TILE_LAYOUT, device=device)
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                wav_dev = gen.inference(mel_dev, frames, 1, sine_noise=noise)
                wav = ttnn.to_torch(wav_dev).float().reshape(-1)
                ttnn.synchronize_device(device)
                times[d].append(time.perf_counter() - t0)
                wavs[d] = wav
                ttnn.deallocate(wav_dev)
                ttnn.deallocate(mel_dev)
        pipe.hift = gens["float32"]
        warm = {d: statistics.median(v[1:]) for d, v in times.items()}
        rows.append((cid, audio_s, {d: v[0] for d, v in times.items()}, warm))
        print(cid, audio_s, {d: [round(x, 3) for x in v] for d, v in times.items()}, flush=True)
    pipe.release()
finally:
    ttnn.close_device(device)
print("| case | audio s | HiFT first sight fp32 / bf16 s | HiFT warm fp32 / bf16 s | warm difference s | RTF difference |")
print("|---|---|---|---|---|---|")
for cid, a, first, warm in rows:
    d = warm["float32"] - warm["bfloat16"]
    print(f"| {cid} | {a:.2f} | {first['float32']:.2f} / {first['bfloat16']:.2f} | {warm['float32']:.3f} / "
          f"{warm['bfloat16']:.3f} | {d:+.3f} | {d / a:+.4f} |")
