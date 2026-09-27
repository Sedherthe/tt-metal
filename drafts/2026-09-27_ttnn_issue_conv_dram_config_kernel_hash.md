<!-- DRAFT, not filed. For tenstorrent/tt-metal, label suggestion: ttnn, conv, performance. -->

# [ttnn][conv][halo] `config_tensors_in_dram=True` puts DRAM buffer addresses in compile-time args, so compiled kernels aren't reused across processes

## Summary

With `Conv1dConfig` / `Conv2dConfig(config_tensors_in_dram=True)`, the conv reader kernels and the halo
(`untilize_with_halo`) reader kernels receive their config tensors' **DRAM addresses as compile-time arguments**:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp:871`:
  `reader_compile_time_args.push_back(conv_reader_indices_buffer->address());  // smuggled-rta-ok: compile-time workload-owned buffer`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.cpp:562`: the same pattern.
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp:307-316`: the padding
  and gather config buffer addresses. The comment there notes this is correct on a program-cache hit because the
  buffers aren't reallocated.

Within one process that is correct. But a compile-time argument is part of the JIT build hash, so the on-disk kernel
cache (`~/.cache/tt-metal-cache`) serves a **new process** only if it allocates those config tensors at exactly the
same DRAM addresses. Any change in allocation order recompiles every conv and halo kernel at every geometry: a
different load order, one extra tensor, a different sequence of inputs. With the config tensors in L1 (the default),
the same conv is reused after the same shift. The pattern is still on `main` as of 2026-09-27.

## Minimal reproducer

```python
# repro.py -- run each invocation in a FRESH process (the in-process program cache would hide the effect)
import glob, os, sys, time, torch, ttnn

CACHE = os.path.expanduser("~/.cache/tt-metal-cache")
C, K, L = 128, 11, 4000  # Conv1d(128 -> 128, k=11)

def binaries():
    return set(glob.glob(os.path.join(CACHE, "*", "kernels", "*", "*", "*", "*.elf")))

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
    before = binaries(); t0 = time.perf_counter()
    ttnn.conv1d(input_tensor=x, weight_tensor=w, device=device, in_channels=C, out_channels=C, batch_size=1,
                input_length=L, kernel_size=K, stride=1, padding=K // 2, conv_config=cfg, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)
    new = binaries() - before
    print(f"config in {where}, {shift_mib} MiB first: {time.perf_counter() - t0:.2f} s, {len(new)} new binaries",
          sorted({p.split(os.sep)[-4] for p in new}))
finally:
    ttnn.close_device(device)
```

Results, one fresh process per line, in this order:

| command | conv1d time | new kernel binaries |
|---|---|---|
| `python repro.py dram 0` | 3.19 s | 21 (first compile) |
| `python repro.py dram 0` | 0.06 s | **0** (same allocation order: reused) |
| `python repro.py dram 1` | 1.46 s | **4**: `halo_gather`, `reader_conv_activations_padded_with_halo_3x3_weights_v2` |
| `python repro.py l1 0` | 1.46 s | 8 (first compile of the L1 variant) |
| `python repro.py l1 1` | 0.06 s | **0** (L1: the same shift compiles nothing) |

Only the kernels whose compile-time arguments carry a config-tensor address recompile. Everything else in the conv
(`conv_bmm_tilize`, the weight readers, `pack_untilize`) is reused.

## Impact

Observed in the CosyVoice2 bring-up (`models/experimental/cosyvoice2`, PR #56651).
- **Why DRAM:** its HiFT vocoder runs about 80 conv1d per utterance length, all with config tensors in DRAM. In L1
  they accumulated in L1_SMALL, one set per geometry; CosyVoice1 hit the same problem.
- **Measured:** three fresh processes each built the model and then ran the same fixed two-call warm-up (N150).
  - The first compiled 2,373 binaries in 566.5 s.
  - The identical second process compiled **0**, in 46.1 s.
  - A third, identical except for a 1 MiB tensor allocated first, recompiled **1,132** binaries, all `halo_gather`
    plus the two conv reader kernels, in 464.6 s.
- **Consequence:** for a conv-heavy model the persistent kernel cache helps only when every process allocates in
  exactly the same order. Kernel prewarm schemes that replay captured compile recipes (e.g. #55465) would inherit
  the problem, since a recipe captured in one process carries that process's addresses.

## Possible fix

Pass the config buffer address as a runtime argument (a `Buffer*` binding patched on cache hit, as the descriptor
factories do elsewhere) and keep only layout facts (page size, accessor args) compile-time. The binary would then
depend on the geometry, not on where the config tensor happens to be allocated.

## Environment

- Wormhole N150 (n150), KMD 2.3.0, firmware bundle 19.11.0, Linux 5.15, host AMD EPYC 7352.
- tt-metal `1f29f312fa` (main, 2026-09-09). The cited lines are unchanged on `main` as of 2026-09-27.
