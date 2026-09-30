"""The "you" clip, device side: streaming's final HiFT call end-padded with silence (as built) against the same call at
its exact length (upstream's semantics: the convs' own zero padding at the true end). No PR code is changed: the exact
variant sets `streaming.FINAL_CALL_BUCKETS = ()` in this process, so the final call runs at its own length. Its new
geometries compile and get their first-sight conv checks; no trace is involved (stage A).

In one process, after the demo's warm-up sequence (so the kernel cache hits):
- all six cases, both variants, with R5's default noise (a generator seeded 1987, llm seed + 1): `padded_1987/` and
  `exact_1987/`, run dirs `scripts/eval_wer_sim.py` scores. `padded_1987` must reproduce R5's audio bit for bit;
- 260-123440-0010, both variants, over nine more noise seeds and upstream's own noise draws: `you_sweep/`;
- the mechanism, all six cases, both variants: upstream's mel, F0 and noise through `HiFTStream`, so only the final
  call's padding differs from upstream (`mech/`). Its tail against upstream's is in `summary.json`.

    python you_tail_ab.py --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import soundfile
import torch

import ttnn

from models.experimental.cosyvoice2.tt import streaming as st
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN
from models.experimental.cosyvoice2.tt.prompt import PromptContext

INPUTS = "/home/user/data/cosyvoice2_inputs"
REF = "/home/user/data/cosyvoice2_streaming_ref"
R5 = "/home/user/data/cosyvoice2_runs/0929/r5_stream_warm1"
R3 = "/home/user/data/cosyvoice2_runs/0929/stream_offline"
YOU = "zero_shot_260-123440-0010"
SEEDS = [1987, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SR = 24000


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(np.square(x, dtype=np.float64))), 1e-12)))


def tail(a: np.ndarray, up: np.ndarray) -> dict:
    n4, n1 = int(0.4 * SR), int(0.02 * SR)
    big = np.nonzero(np.abs(a) >= 1e-4)[0]
    return {"last_0.4s_signal_dbfs": round(dbfs(up[-n4:]), 1), "last_0.4s_diff_dbfs": round(dbfs(a[-n4:] - up[-n4:]), 1),
            "last_20ms_dbfs": round(dbfs(a[-n1:]), 1), "upstream_last_20ms_dbfs": round(dbfs(up[-n1:]), 1),
            "quiet_samples_at_end": int(len(a) - 1 - big[-1]) if len(big) else len(a)}  # fmt: skip


def seeded(seed: int, harmonics: int):
    gen = torch.Generator().manual_seed(seed)
    return lambda k, samples: torch.randn(1, samples, harmonics, generator=gen)


def upstream_noise(ref):
    return lambda k, samples: torch.from_numpy(ref[f"hift_noise_{k}"][:, :samples])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = sorted(os.path.basename(p)[: -len(".npz")] for p in os.listdir(REF) if p.endswith(".npz"))
    cases = [c for c in cases if c.startswith("zero_shot_")]
    for sub in ("padded_1987", "exact_1987", "you_sweep", "mech"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)  # the demo's
    summary = {"sanity": {}, "mech": {}, "you_sweep": {}, "corpus": {}}
    try:
        pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        dtype = getattr(ttnn, pipe.config.hift_source_dtype)
        results = {"padded": [], "exact": []}
        for variant, buckets in (("padded", (128, 256)), ("exact", ())):
            st.FINAL_CALL_BUCKETS = buckets
            for c in cases:
                ref = np.load(f"{REF}/{c}.npz")
                ctx = PromptContext.from_npz(f"{INPUTS}/{c}.npz")
                tokens = ref["tokens"].tolist()
                up = ref["audio"]
                # our whole stage A with R5's default noise
                audio = np.concatenate([x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, seeded(1987, pipe.harmonics))])
                audio = audio.astype(np.float32)
                soundfile.write(f"{args.out}/{variant}_1987/{c}.wav", audio, SR)
                results[variant].append({**json.loads(str(np.load(f"{INPUTS}/{c}.npz")["case_json"])), "wav": f"{c}.wav",
                                         "audio_s": round(len(audio) / SR, 3), "segment_tokens": [tokens]})  # fmt: skip
                summary["corpus"][f"{variant}/{c}"] = tail(audio, up)
                if variant == "padded":
                    r5 = soundfile.read(f"{R5}/{c}.wav", dtype="float32")[0]
                    summary["sanity"][c] = {"equals_r5": bool(np.array_equal(audio, r5)),
                                            "sha1": hashlib.sha1(audio.tobytes()).hexdigest()[:12]}  # fmt: skip
                # the mechanism: upstream's mel, F0 and noise; only the final call's padding differs from upstream
                mech = st.HiFTStream(pipe.hift, pipe.harmonics, dtype=dtype)
                sched = st.stream_schedule(len(tokens), ctx.n_prompt_tokens)
                got = [mech.step(torch.from_numpy(ref[f"mel_{k}"]), ch.final, torch.from_numpy(ref[f"hift_noise_{k}"]),
                                 f0=torch.from_numpy(ref[f"hift_f0_{k}"])) for k, ch in enumerate(sched)]  # fmt: skip
                m = np.concatenate(got).astype(np.float32)
                soundfile.write(f"{args.out}/mech/{variant}_{c}.wav", m, SR)
                summary["mech"][f"{variant}/{c}"] = tail(m, up)
                print(f"{variant} {c}: ours {summary['corpus'][f'{variant}/{c}']}; mech {summary['mech'][f'{variant}/{c}']}", flush=True)
                if c == YOU:
                    for s in SEEDS[1:]:
                        a = np.concatenate([x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, seeded(s, pipe.harmonics))])
                        soundfile.write(f"{args.out}/you_sweep/{variant}_seed{s}.wav", a.astype(np.float32), SR)
                        summary["you_sweep"][f"{variant}/seed{s}"] = tail(a, up)
                    a = np.concatenate([x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, upstream_noise(ref))])
                    a = a.astype(np.float32)
                    soundfile.write(f"{args.out}/you_sweep/{variant}_upnoise.wav", a, SR)
                    summary["you_sweep"][f"{variant}/upnoise"] = tail(a, up)
                    if variant == "padded":
                        r3 = soundfile.read(f"{R3}/{c}.wav", dtype="float32")[0]
                        summary["sanity"]["upnoise_equals_r3"] = bool(np.array_equal(a, r3))
        for variant, rows in results.items():
            with open(f"{args.out}/{variant}_1987/results.json", "w") as fh:
                json.dump({"backend": f"ttnn-streaming-offline-{variant}-final", "results": rows}, fh, indent=2)
    finally:
        pipe.release() if "pipe" in locals() else None
        ttnn.close_device(device)
    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["sanity"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
