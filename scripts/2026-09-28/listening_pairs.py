"""HiFT tail listening pairs: for a few Stage 1 utterances, the same tokens -> the same (bucketed) flow mel -> HiFT
twice, with one fixed sine-noise draw: once at its bucket (silence-padded, trimmed; what the pipeline ships) and once
at its exact length (the unbucketed path). Only HiFT's padding differs between the two files of a pair.

    python listening_pairs.py <stage 1 demo dir> <case,case,...> <out dir>
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import soundfile
import torch

import ttnn
from models.experimental.cosyvoice2.tt.pipeline import SAMPLE_RATE, CosyVoice2TTNN, bucket_at_least
from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources

run_dir, cases, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
inputs = os.environ.get("COSYVOICE2_INPUTS", "/home/user/data/cosyvoice2_inputs")
os.makedirs(out, exist_ok=True)
with open(os.path.join(run_dir, "results.json")) as fh:
    results = {r["case_id"]: r for r in json.load(fh)["results"]}

device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)
rows = []
try:
    pipe = CosyVoice2TTNN(device)
    bucketed_cfg = pipe.config
    for case_id in cases:
        r = results[case_id]
        assert len(r["segment_tokens"]) == 1, "single-segment cases only"
        ctx = PromptContext.from_npz(os.path.join(inputs, f"{case_id}.npz"))
        mel = pipe.tokens_to_mel(r["segment_tokens"][0], ctx)
        frames = int(mel.shape[1])
        bucket = bucket_at_least(frames, bucketed_cfg.hift_frame_buckets())
        noise = torch.randn(1, frames * pipe.hift.upsample_scale, pipe.harmonics, generator=torch.Generator().manual_seed(0))
        wavs = {}
        for name, cfg in (("bucketed", bucketed_cfg), ("exact", replace(bucketed_cfg, bucketing=False))):
            pipe.config = cfg
            wavs[name] = pipe.mel_to_wav(mel, RandomSources(sine_noise=noise)).numpy().astype(np.float32)
            soundfile.write(os.path.join(out, f"{case_id}_{name}.wav"), wavs[name], SAMPLE_RATE)
        pipe.config = bucketed_cfg
        d = wavs["bucketed"] - wavs["exact"]
        n = len(d)
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
        above = np.nonzero(np.abs(d) > 1e-3)[0]
        rows.append(
            {
                "case_id": case_id,
                "audio_s": round(n / SAMPLE_RATE, 3),
                "mel_frames": frames,
                "hift_bucket": bucket,
                "pad_frames": bucket - frames,
                "reach_ms": round((n - above[0]) / SAMPLE_RATE * 1000) if len(above) else 0,
                "last_400ms_rms_diff_db_rel_signal": round(
                    20 * np.log10(rms(d[-9600:]) / max(rms(wavs["exact"][-9600:]), 1e-9)), 1
                ),
                "max_abs_diff_last_400ms": round(float(np.abs(d[-9600:]).max()), 4),
                "max_abs_diff_before_last_500ms": round(float(np.abs(d[:-12000]).max()), 6),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    pipe.release()
finally:
    ttnn.close_device(device)
with open(os.path.join(out, "pairs.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
