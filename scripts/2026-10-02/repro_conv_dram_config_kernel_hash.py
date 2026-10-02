"""Minimal reproducer: with Conv1dConfig(config_tensors_in_dram=True), ttnn.conv1d's reader kernel takes the config
tensor's DRAM address as a compile-time argument, so the kernel binary's hash depends on where that tensor was
allocated. A fresh process that allocates in a different order recompiles the same conv at the same shape instead of
reusing the binary already in the on-disk JIT cache.

Run each line in a FRESH process (the in-process program cache would hide the effect):
    python repro_conv_dram_config_kernel_hash.py dram 0   # first: compiles
    python repro_conv_dram_config_kernel_hash.py dram 0   # same allocation order: 0 new binaries
    python repro_conv_dram_config_kernel_hash.py dram 1   # 1 MiB allocated first: recompiles the reader kernel
    python repro_conv_dram_config_kernel_hash.py l1 0     # config tensors in L1 (the default)
    python repro_conv_dram_config_kernel_hash.py l1 1     # L1: shifting DRAM allocation compiles nothing new
"""
import glob
import os
import sys
import time

import torch

import ttnn

# 2026-10-02: the cache root to scan is TT_METAL_CACHE when set (a fresh one gives clean counts); otherwise as before
CACHE = os.environ.get("TT_METAL_CACHE") or os.path.expanduser("~/.cache/tt-metal-cache")
C, K, L = 128, 11, 4000  # a HiFT-resblock-like Conv1d(128 -> 128, k=11)


def binaries():
    return set(glob.glob(os.path.join(CACHE, "**", "kernels", "*", "*", "*", "*.elf"), recursive=True))


where, shift_mib = sys.argv[1], int(sys.argv[2])
device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    pad = None
    if shift_mib:  # shifts every later DRAM allocation, the conv's config tensor included
        pad = ttnn.from_torch(torch.zeros(1, 1, shift_mib * 2**20 // 64, 32), dtype=ttnn.bfloat16, device=device)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, L, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(torch.randn(C, C, K) * 0.05, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=(where == "dram"))
    before = binaries()
    t0 = time.perf_counter()
    out = ttnn.conv1d(input_tensor=x, weight_tensor=w, device=device, in_channels=C, out_channels=C, batch_size=1,
                      input_length=L, kernel_size=K, stride=1, padding=K // 2, conv_config=cfg, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)
    secs = time.perf_counter() - t0
    new = binaries() - before
    kernels = sorted({p.split(os.sep)[-4] for p in new})
    print(f"config tensors in {where.upper()}, {shift_mib} MiB allocated first: conv1d {secs:.2f} s, "
          f"{len(new)} new kernel binaries {kernels}")
finally:
    ttnn.close_device(device)
