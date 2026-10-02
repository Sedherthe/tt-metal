<!-- DRAFT, not filed; for the user to file on tenstorrent/tt-metal (labels: ttnn, conv, performance). It replaces
2026-09-27_ttnn_issue_conv_dram_config_kernel_hash.md:
- the reproducer was re-run on 2026-10-02 on a second N150 (KMD 2.9.0), against an empty kernel cache;
- the cited lines were checked on that day's upstream main (6a39ae0a22);
- the smuggled-rta-ok context (#49141) and the Buffer* precedent (#49301) are added.
The "Impact" figures are from 2026-09-27 and were not re-measured. Script and log:
scripts/2026-10-02/repro_conv_dram_config_kernel_hash.py, phase_repro.sh. -->

# [ttnn][conv][halo] `config_tensors_in_dram=True` puts DRAM buffer addresses in compile-time args, so compiled kernels aren't reused across processes

## Summary

With `Conv1dConfig` / `Conv2dConfig(config_tensors_in_dram=True)`, two kinds of reader kernel get their config
tensors' **DRAM addresses as compile-time arguments**: the conv reader kernels and the halo (`untilize_with_halo`)
reader kernels. On `main` (`6a39ae0a22`, 2026-10-02):

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp:871`:
  `conv_reader_indices_buffer->address());  // smuggled-rta-ok: compile-time workload-owned buffer`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.cpp:562`:
  `activation_kernel_compile_args.push_back(conv_reader_indices_buffer->address());  // smuggled-rta-ok`
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp:307-316`: the padding
  and gather config buffer addresses, each marked `// smuggled-rta-ok: see NOTE above`.

Within one process this is correct, as the NOTE there says: the buffers are owned by the cached workload and never
reallocated, so the baked address stays valid on a program-cache hit. That is why #49141's guard is exempted here.

But a compile-time argument is part of the JIT build hash. So the on-disk kernel cache serves a **new process** only
if that process allocates the config tensors at exactly the same DRAM addresses. Any change in allocation order
recompiles every conv and halo kernel at every geometry: a different load order, one extra tensor, or a different
sequence of inputs. With the config tensors in L1 (the default), the same conv is reused after the same shift.

## Minimal reproducer

```python
# repro.py -- run each invocation in a FRESH process (the in-process program cache would hide the effect).
# For clean counts, point TT_METAL_CACHE at an empty directory first.
import glob, os, sys, time, torch, ttnn

CACHE = os.environ.get("TT_METAL_CACHE") or os.path.expanduser("~/.cache/tt-metal-cache")
C, K, L = 128, 11, 4000  # Conv1d(128 -> 128, k=11)

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

The results, with one fresh process per line in this order and `TT_METAL_CACHE` an empty directory:

| command | conv1d time | new kernel binaries |
|---|---|---|
| `python repro.py dram 0` | 4.56 s | 27 (first compile) |
| `python repro.py dram 0` | 0.06 s | **0** (same allocation order: reused) |
| `python repro.py dram 1` | 1.49 s | **4**: `halo_gather`, `reader_conv_activations_padded_with_halo_3x3_weights_v2` |
| `python repro.py l1 0` | 1.52 s | 8 (first compile of the L1 variant) |
| `python repro.py l1 1` | 0.07 s | **0** (L1: the same shift compiles nothing) |

- Only the kernels whose compile-time arguments carry a config-tensor address recompile. The rest of the conv
  (`conv_bmm_tilize`, the weight readers, `pack_untilize`) is reused.
- A second N150 gave the same verdicts on 2026-09-27: 0 and 4 binaries on the second and third lines, 8 and 0 on
  the last two. Its first line compiled 21, because a few generic kernels were already in its cache.

## Impact

Observed in the CosyVoice2 bring-up (`models/experimental/cosyvoice2`, #56651).
- **Why DRAM:** its HiFT vocoder runs about 80 conv1d per utterance length, all with config tensors in DRAM. In L1
  they accumulated in L1_SMALL, one set per geometry; CosyVoice1 hit the same problem.
- **Measured (2026-09-27, N150):** three fresh processes each built the model and then ran the same fixed two-call
  warm-up.
  - The first compiled 2,373 binaries in 566.5 s.
  - The identical second process compiled **0**, in 46.1 s.
  - A third, identical except for a 1 MiB tensor allocated first, recompiled **1,132** binaries, all `halo_gather`
    plus the two conv reader kernels, in 464.6 s.
- **Consequence:** for a conv-heavy model the persistent kernel cache helps only when every process allocates in
  exactly the same order.
  - The model works around it with a fixed warm-up sequence at start-up. It takes 3.1 min with the kernels on
    disk, and 31.4 min on an empty cache.
  - Kernel-prewarm schemes that replay captured compile recipes (for example the open #55465) would inherit the
    problem, since a recipe captured in one process carries that process's addresses.

## Possible fix

Pass the config buffer address as a runtime argument instead. A `Buffer*` binding, which the framework patches on a
program-cache hit, as #49301 did for the unary op, would keep the in-process behaviour. Only layout facts (page
size, accessor args) would stay compile-time. The binary would then depend on the geometry, not on where the config
tensor happens to be allocated.

## Environment

- **2026-10-02:** Wormhole N150 (n150 L), KMD 2.9.0, firmware bundle 19.11.0, Linux 6.8, host AMD EPYC 7352.
- **2026-09-27:** another N150, KMD 2.3.0, firmware bundle 19.11.0, Linux 5.15.
- **Code:** tt-metal `1f29f312fa` (main, 2026-09-09) on both boards. The cited lines are unchanged on `main` at
  `6a39ae0a22` (2026-10-02).
