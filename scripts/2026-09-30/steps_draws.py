"""Stage 3 plan, step 3, the device half: the CFM's Euler step count swept, WER/SIM material over D43's five noise
draws, Stage 1 and streaming, with the timings.

`scripts/noise_draws.py`'s protocol (the reported configuration, both warm-ups, the demo's tokens at LLM seed 1986,
the vocoder's noise from `RandomSources.noise_seed`), with one change: the flow's step count, the module constant
`tt/flow/flow.py:N_TIMESTEPS` that both `inference` and `inference_streaming` read at call time, set in this process
only. The PR is unchanged. Upstream hardcodes 10 (`CausalMaskedDiffWithXvec.inference`), and so does the port.

One process: build, warm up once, then each step count in turn (10 first, which must reproduce D43's 10-step draws
exactly), each over the noise seeds, Stage 1 then streaming. It writes
`<out>/steps<k>/tt_{stage1,stream}_seed<N>/` (wavs and results.json in `eval_wer_sim.py`'s layout, with the timings:
RTF, the stage totals, and for streaming the first audio and the per-chunk records). It checks that every run
sampled D43's tokens.

    python steps_draws.py --out <dir> [--steps 10,8,6,5] [--noise-seeds 1,2,3,4,5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile

import ttnn
from models.experimental.cosyvoice2.tt.flow import flow as flow_mod
from models.experimental.cosyvoice2.tt.pipeline import SAMPLE_RATE, CosyVoice2Config, CosyVoice2TTNN
from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources

INPUTS = "/home/user/data/cosyvoice2_inputs"
D43 = "/home/user/data/cosyvoice2_draws"  # D43's 10-step TT draws: tt_{stage1,stream}_seed<N>


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="10,8,6,5")
    ap.add_argument("--noise-seeds", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=1986)
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in args.noise_seeds.split(",")]
    ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(INPUTS, "*.npz")))]
    ctxs = [c for c in ctxs if c.meta["case"]["set"] == "librispeech"]
    d43_tokens = {r["case_id"]: r["segment_tokens"] for r in json.load(open(f"{D43}/tt_stage1_seed1/results.json"))["results"]}
    assert flow_mod.N_TIMESTEPS == 10

    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)  # the demo's
    same_as_d43 = []
    try:
        pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
        t0 = time.perf_counter()
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        print(f"warm-ups {time.perf_counter() - t0:.1f} s", flush=True)
        for k in steps:
            flow_mod.N_TIMESTEPS = k
            for seed in seeds:
                for mode in ("stage1", "stream"):
                    out_dir = os.path.join(args.out, f"steps{k}", f"tt_{mode}_seed{seed}")
                    os.makedirs(out_dir, exist_ok=True)
                    synth = pipe.synthesize if mode == "stage1" else pipe.synthesize_stream
                    results = []
                    for ctx in ctxs:
                        case = ctx.meta["case"]
                        syn = synth(ctx, case["text"], rng=RandomSources(llm_seed=args.seed, noise_seed=seed))
                        assert not pipe.live_traces(), pipe.live_traces()
                        name = f"{case['case_id']}.wav"
                        soundfile.write(os.path.join(out_dir, name), syn.audio, SAMPLE_RATE)
                        tokens = [list(map(int, s.tokens)) for s in syn.segments]
                        assert tokens == d43_tokens[case["case_id"]], f"{case['case_id']}: not D43's tokens"
                        if k == 10:  # D43's draw of the same seed and mode, read back from its wav
                            mine = soundfile.read(os.path.join(out_dir, name), dtype="int16")[0]
                            theirs = soundfile.read(os.path.join(D43, f"tt_{mode}_seed{seed}", name), dtype="int16")[0]
                            same_as_d43.append(bool(mine.shape == theirs.shape and np.array_equal(mine, theirs)))
                        row = {**case, "wav": name, "audio_s": round(syn.audio_s, 3), "wall_s": round(syn.wall_s, 4),
                               "rtf": round(syn.rtf, 4), "n_timesteps": k, "noise_seed": seed, "segment_tokens": tokens,
                               "stage_s": {n: round(v, 4) for n, v in syn.stage_totals().items()}}  # fmt: skip
                        if mode == "stream":
                            row.update(first_audio_s=round(syn.first_audio_s, 4), chunks=syn.chunks)
                        results.append(row)
                        extra = f" first audio {syn.first_audio_s:.3f} s" if mode == "stream" else ""
                        print(f"  steps {k:2d} seed {seed} {mode:6s} {case['case_id']:<34} audio {syn.audio_s:6.2f} s "
                              f"RTF {syn.rtf:.3f}{extra}", flush=True)  # fmt: skip
                    with open(os.path.join(out_dir, "results.json"), "w") as fh:
                        json.dump({"backend": f"ttnn-{mode}", "llm_seed": args.seed, "noise_seed": seed, "n_timesteps": k,
                                   "results": results}, fh, indent=2, ensure_ascii=False)  # fmt: skip
    finally:
        flow_mod.N_TIMESTEPS = 10
        if "pipe" in locals():
            pipe.release()
        ttnn.close_device(device)
    if same_as_d43:
        print(f"10 steps against D43's draws: {sum(same_as_d43)} of {len(same_as_d43)} wavs identical", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
