"""The bucketed pipeline's start-up warm-up, measured in a fresh process: every bucket's time, the device memory the
warmed set leaves (DRAM against the conv cache's old 150 MB free-DRAM eviction threshold; L1_SMALL), how many kernels
the process compiled, and where the per-geometry first-sight time goes (item d): the conv safety check
(`_verify_and_resolve`), weight preparation (`_prepared`), everything else.

    python warmup_measure.py --label dram-1            # config tensors in DRAM (the design under test)
    python warmup_measure.py --label l1 --l1 --l1-small-kib 1024
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
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN, device_memory

# TT_METAL_CACHE=<dir> puts the kernel cache at <dir>/tt-metal-cache<build key>/kernels (no separator before the
# key); the default is ~/.cache/tt-metal-cache/<build key>/kernels.
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

# Eviction exposure: every GeometryWeightCache.put() checks free DRAM per bank against its threshold (0 in bucketed
# mode). Record what each check saw, to compare with the default 150 MB threshold.
from models.experimental.cosyvoice2.tt import geometry_cache as gc_mod

DEFAULT_THRESHOLD_B = gc_mod._DEFAULT_THRESHOLD_MB * 1024 * 1024
put_free = []
_free = gc_mod.GeometryWeightCache._free_bytes_per_bank


def recording_free(self):
    v = _free(self)
    put_free.append(v)
    return v


gc_mod.GeometryWeightCache._free_bytes_per_bank = recording_free

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--l1", action="store_true", help="config tensors in L1 instead of DRAM")
ap.add_argument("--l1-small-kib", type=int, default=64)
ap.add_argument("--stop-at-l1-small-kib", type=int, default=None, help="end the warm-up once L1_SMALL passes this")
ap.add_argument("--out", help="also write the results as JSON here")
args = ap.parse_args()

before = binaries()
t_proc = time.perf_counter()
device = ttnn.open_device(device_id=0, l1_small_size=args.l1_small_kib * 1024, trace_region_size=50_000_000)
try:
    cfg = CosyVoice2Config(conv_config_tensors_in_dram=not args.l1)
    t0 = time.perf_counter()
    pipe = CosyVoice2TTNN(device, cfg)
    build_s = time.perf_counter() - t0
    mem_build = device_memory(device)
    t0 = time.perf_counter()
    per_geometry_mem = {}
    try:
        def on_geometry(name):
            per_geometry_mem[name] = {
                **device_memory(device),
                "t": time.perf_counter(),
                "check_s": spent["conv_safety_check"],
                "prep_s": spent["conv_weight_prep"],
                "binaries": len(binaries()),
            }
            if args.stop_at_l1_small_kib and per_geometry_mem[name]["l1_small"] > args.stop_at_l1_small_kib * 1024:
                raise RuntimeError(f"L1_SMALL passed {args.stop_at_l1_small_kib} KiB after {name}")

        clock = pipe.warmup_buckets(on_geometry=on_geometry)
        error = None
    except Exception as e:  # noqa: BLE001 -- report how far it got (e.g. L1_SMALL exhausted)
        clock, error = {}, f"{type(e).__name__}: {str(e)[:400]}"
    warm_s = time.perf_counter() - t0
    mem = device_memory(device)
    free = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_free_per_bank
    l1s = ttnn.get_memory_view(device, ttnn.BufferType.L1_SMALL)
    evictions = pipe.conv_cache_evictions()
    pipe.release()
finally:
    ttnn.close_device(device)
new = binaries() - before
groups = collections.defaultdict(float)
for k, v in clock.items():
    groups[k.split("_")[0]] += v
summary = {
    "label": args.label, "config_tensors": "L1" if args.l1 else "DRAM", "l1_small_kib": args.l1_small_kib,
    "kernel_cache": CACHE, "error": error, "build_s": round(build_s, 1), "warmup_s": round(warm_s, 1),
    "warmup_by_component_s": {k: round(v, 1) for k, v in groups.items()},
    "conv_safety_check_s": round(spent["conv_safety_check"], 1), "conv_weight_prep_s": round(spent["conv_weight_prep"], 1),
    "dram_mib_per_bank": {"after_build": round(mem_build["dram"] / 2**20, 1), "after_warmup": round(mem["dram"] / 2**20, 1),
                          "free_after_warmup": round(free / 2**20, 1)},
    "conv_cache_puts": len(put_free),
    "min_free_dram_mib_per_bank_at_put": round(min(put_free) / 2**20, 1) if put_free else None,
    "puts_below_default_threshold": sum(v < DEFAULT_THRESHOLD_B for v in put_free),
    "l1_small_bytes_per_bank": {"allocated": mem["l1_small"], "free": l1s.total_bytes_free_per_bank},
    "conv_cache_evictions": evictions, "kernels_compiled": len(new), "process_s": round(time.perf_counter() - t_proc, 1),
}
print(json.dumps(summary))
print(json.dumps({"per_bucket_s": {k: round(v, 2) for k, v in clock.items()}}))
prev = None
rows = {}
for k, v in per_geometry_mem.items():
    rows[k] = {
        "s": None if prev is None else round(v["t"] - prev["t"], 2),
        "check_s": round(v["check_s"] - (0 if prev is None else prev["check_s"]), 2),
        "prep_s": round(v["prep_s"] - (0 if prev is None else prev["prep_s"]), 2),
        "new_binaries": v["binaries"] - (len(before) if prev is None else prev["binaries"]),
        "dram_mib": round(v["dram"] / 2**20, 2), "l1_small_b": v["l1_small"],
    }
    prev = v
print(json.dumps({"per_geometry": rows}))
if args.out:
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "per_bucket_s": clock, "per_geometry": rows}, fh, indent=1)
