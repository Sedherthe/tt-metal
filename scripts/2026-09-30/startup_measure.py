"""R6: start-up on the masked HiFT, cold and warm. One fresh process: pipeline construction, `warmup_buckets()` and
`warmup_streaming()`, each timed, with the kernel binaries the process compiled and the conv safety checks' and
weight preparation's share. Run it twice against the same new `TT_METAL_CACHE` directory: the first starts from an
empty kernel cache, the second is identical (09-28's protocol, scripts/2026-09-28/warmup_measure.py, plus the
streaming set that `demo.py --stream` warms).

    TT_METAL_CACHE=<new dir> python startup_measure.py --label cold --out <json>
"""
import argparse
import collections
import glob
import json
import os
import time

import ttnn
from models.experimental.cosyvoice2.tt.hifigan import conv as conv_mod
from models.experimental.cosyvoice2.tt.hifigan import upsample as up_mod
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN

# TT_METAL_CACHE=<dir> puts the kernel cache at <dir>/tt-metal-cache<build key>/kernels (no separator before the key)
CACHE = (
    os.path.join(os.environ["TT_METAL_CACHE"], "tt-metal-cache*")
    if os.environ.get("TT_METAL_CACHE")
    else os.path.join(os.path.expanduser("~/.cache/tt-metal-cache"), "*")
)
spent = collections.Counter()


def binaries():
    return set(glob.glob(os.path.join(CACHE, "kernels", "*", "*", "*", "*.elf")))


def timed_method(cls, name, key):
    original = getattr(cls, name)

    def wrapper(self, *a, **k):
        ttnn.synchronize_device(self.device)
        t0 = time.perf_counter()
        try:
            return original(self, *a, **k)
        finally:
            ttnn.synchronize_device(self.device)
            spent[key] += time.perf_counter() - t0

    setattr(cls, name, wrapper)


for cls in (conv_mod.TtConv1d, up_mod.TtConvTranspose1d):
    timed_method(cls, "_verify_and_resolve", "conv_safety_check")
    timed_method(cls, "_prepared", "conv_weight_prep")

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

before = binaries()
device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)  # the demo's
try:
    t0 = time.perf_counter()
    pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
    build_s = time.perf_counter() - t0
    phases = {}
    for name, fn in (("warmup_buckets", pipe.warmup_buckets), ("warmup_streaming", pipe.warmup_streaming)):
        n0, c0, p0 = len(binaries()), spent["conv_safety_check"], spent["conv_weight_prep"]
        t0 = time.perf_counter()
        clock = fn()
        wall = time.perf_counter() - t0
        groups = collections.defaultdict(float)
        for k, v in clock.items():
            groups[k.split("_")[0] if name == "warmup_buckets" else k.split("_")[0] + "_" + k.split("_")[1]] += v
        phases[name] = {"s": round(wall, 1), "binaries": len(binaries()) - n0, "by_stage_s": {k: round(v, 1) for k, v in groups.items()},
                        "conv_safety_check_s": round(spent["conv_safety_check"] - c0, 1), "conv_weight_prep_s": round(spent["conv_weight_prep"] - p0, 1)}  # fmt: skip
        print(f"[{args.label}] {name}: {phases[name]}", flush=True)
    pipe.release()
finally:
    ttnn.close_device(device)
summary = {"label": args.label, "kernel_cache": CACHE, "build_s": round(build_s, 1), **phases,
           "binaries_compiled": len(binaries() - before)}  # fmt: skip
with open(args.out, "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary), flush=True)
