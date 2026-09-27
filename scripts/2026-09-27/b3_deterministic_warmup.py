"""B3 / item 4a: does a fixed, deterministic warm-up sequence at process start let a second fresh process reuse the
first one's compiled kernels from the disk cache?

Each run, in a fresh process: open the device the demo's way, optionally allocate a dummy DRAM tensor first (the
negative control: same sequence, allocation shifted), build CosyVoice2TTNN, then run the fixed warm-up: two
sentences with fixed seeds on one fixed prompt. Reports per-call stage times and how many kernel binaries the
process compiled (new *.elf files in the tt-metal kernel cache).

    python b3_deterministic_warmup.py --label p1
    python b3_deterministic_warmup.py --label p2
    python b3_deterministic_warmup.py --label p3 --perturb-mib 1
"""
import argparse
import glob
import json
import os
import time

import torch

import ttnn
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN, device_memory
from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources

CACHE = os.path.expanduser("~/.cache/tt-metal-cache")
PROMPT = "/home/user/data/cosyvoice2_inputs/zero_shot_121-127105-0015.npz"
WARMUP = [  # never synthesized on TT before this experiment: their lengths are new to the kernel cache
    ("w1", "The quick brown fox jumps over the lazy dog while the morning sun rises slowly.", 11),
    ("w2", "A small boat drifted past the harbour lights just before the evening tide came in.", 22),
]


def elfs():
    return set(glob.glob(os.path.join(CACHE, "*", "kernels", "*", "*", "*", "*.elf")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--perturb-mib", type=int, default=0)
    args = ap.parse_args()

    before = elfs()
    t_proc = time.perf_counter()
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)
    rows, dummy = [], None
    try:
        if args.perturb_mib:
            n = args.perturb_mib * 2**20 // 2  # bf16 elements
            dummy = ttnn.from_torch(torch.zeros(1, 1, n // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        t0 = time.perf_counter()
        pipe = CosyVoice2TTNN(device)
        build_s = time.perf_counter() - t0
        mem_after_build = device_memory(device)
        ctx = PromptContext.from_npz(PROMPT)
        for name, text, seed in WARMUP:
            e0 = len(elfs())
            syn = pipe.synthesize(ctx, text, rng=RandomSources(llm_seed=seed))
            t = syn.stage_totals()
            rows.append({"call": name, "tokens": len(syn.tokens), "audio_s": round(syn.audio_s, 2), "wall_s": round(syn.wall_s, 2),
                         "llm_s": round(t["llm_prefill"] + t["llm_decode"], 2), "flow_s": round(t["flow_encoder"] + t["flow_cfm"], 2),
                         "hift_s": round(t["hift"], 2), "new_elfs": len(elfs()) - e0})
            print(json.dumps(rows[-1]), flush=True)
        pipe.release()
    finally:
        if dummy is not None:
            ttnn.deallocate(dummy)
        ttnn.close_device(device)
    new = elfs() - before
    kinds = sorted({p.split(os.sep)[-4] for p in new})
    print(json.dumps({"label": args.label, "perturb_mib": args.perturb_mib, "build_s": round(build_s, 1),
                      "dram_after_build": mem_after_build["dram"], "process_s": round(time.perf_counter() - t_proc, 1),
                      "new_elfs_total": len(new), "new_elf_kernels": kinds}))


if __name__ == "__main__":
    main()
