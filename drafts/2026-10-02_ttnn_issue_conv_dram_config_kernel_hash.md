<!-- DRAFT, not filed; for the user to file on tenstorrent/tt-metal with the "Bug Report" form
(.github/ISSUE_TEMPLATE/bug_report.yml). Each "##" heading below is one field of the form, in its order; paste
the text under it into that field. It replaces 2026-09-27_ttnn_issue_conv_dram_config_kernel_hash.md:
- the reproducer was re-run on 2026-10-02 on a second N150 (KMD 2.9.0), against an empty kernel cache;
- the cited lines were checked on that day's upstream main (6a39ae0a22);
- the smuggled-rta-ok context (#49141) and the Buffer* precedent (#49301) are added.
The "Impact" figures are from 2026-09-27 and were not re-measured. Script and log:
scripts/2026-10-02/repro_conv_dram_config_kernel_hash.py, phase_repro.sh. -->

## Title

[ops/conv]: `config_tensors_in_dram=True` bakes DRAM buffer addresses into compile-time args, so compiled kernels aren't reused across processes

## Component / Area

ops: conv1d / conv2d and the sliding-window halo (`untilize_with_halo`); the on-disk JIT kernel cache

## Issue Type (optional)

Other (the on-disk kernel cache misses: performance, not correctness)

## Observed

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
recompiles the conv and halo reader kernels at every geometry: a different load order, one extra tensor, or a
different sequence of inputs. With the config tensors in L1 (the default), the same conv is reused after the same
shift.

The reproducer below, five fresh processes in order against an empty `TT_METAL_CACHE`:

| command | conv1d time | new kernel binaries |
|---|---|---|
| `python repro.py dram 0` | 4.56 s | 27 (first compile) |
| `python repro.py dram 0` | 0.06 s | **0** (same allocation order: reused) |
| `python repro.py dram 1` | 1.49 s | **4**: `halo_gather`, `reader_conv_activations_padded_with_halo_3x3_weights_v2` |
| `python repro.py l1 0` | 1.52 s | 8 (first compile of the L1 variant) |
| `python repro.py l1 1` | 0.07 s | **0** (L1: the same shift compiles nothing) |

Only the kernels whose compile-time arguments carry a config-tensor address recompile. The rest of the conv
(`conv_bmm_tilize`, the weight readers, `pack_untilize`) is reused.

## Expected

A fresh process that runs the same conv, at the same geometry with the same configuration, should reuse the
binaries already in the on-disk kernel cache, wherever its config tensors land in DRAM. That is what happens with
the config tensors in L1. The kernel binary should depend on the geometry and the configuration, not on the config
tensor's address.

In the reproducer, the third run (`dram 1`) should compile **0** new binaries, as `l1 1` does, instead of
recompiling `halo_gather` and the conv reader.

A possible fix: pass the config buffer's address as a runtime argument instead.
- A `Buffer*` binding, which the framework patches on a program-cache hit (as #49301 did for the unary op), would
  keep today's in-process behaviour.
- Only layout facts (page size, accessor args) would stay compile-time.

## 1. Steps (exact commands)

Save this as `repro.py`, then run each line below in a **fresh** process, in this order. The in-process program
cache would hide the effect, and the empty `TT_METAL_CACHE` makes the counts clean.

```bash
export TT_METAL_CACHE=$(mktemp -d)
python repro.py dram 0   # first compile
python repro.py dram 0   # same allocation order: 0 new binaries
python repro.py dram 1   # 1 MiB allocated first: the conv reader and halo kernels recompile
python repro.py l1 0     # config tensors in L1 (the default): first compile of that variant
python repro.py l1 1     # L1: the same 1 MiB shift compiles nothing
```

```python
# repro.py
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
    secs = time.perf_counter() - t0
    new = binaries() - before
    kernels = sorted({p.split(os.sep)[-4] for p in new})
    print(f"config tensors in {where.upper()}, {shift_mib} MiB allocated first: conv1d {secs:.2f} s, "
          f"{len(new)} new kernel binaries {kernels}")
finally:
    ttnn.close_device(device)
```

## 2. Input data / link or description

None. The reproducer makes its own random input and weight (`torch.manual_seed(0)`): one `Conv1d(128 -> 128,
k=11)` over a length of 4,000. No model, checkpoint or dataset is needed.

## 3. Frequency

Always. It is deterministic: every fresh process whose config tensors land at different DRAM addresses from the
process that filled the cache recompiles those kernels.

It reproduced on two N150 boards: on 2026-09-27 (KMD 2.3.0) and on 2026-10-02 (KMD 2.9.0, empty cache), with the
same verdicts each time (0 / 4 / 8 / 0 new binaries on runs 2–5).

## 1. Software Versions

- OS: Ubuntu 22.04.5 LTS, Linux 6.8.0-124 (2026-10-02); Linux 5.15 (2026-09-27)
- tt-metal: `1f29f312fa` (main, 2026-09-09), built from source (`build_metal.sh`, Release). The cited lines are
  unchanged on `main` at `6a39ae0a22` (2026-10-02).
- KMD 2.9.0 (2026-10-02); 2.3.0 (2026-09-27). Firmware bundle 19.11.0.
- Python 3.10.19, torch 2.11.0+cpu

## 2. Hardware Details

- Product: Wormhole
- Card/System: N150 (n150 L), host AMD EPYC 7352. Reproduced on two separate N150 cards.

## Is this a regression? (optional)

Unknown

## Regression Details (optional)

Not bisected. The pattern is present wherever `config_tensors_in_dram` reaches these program factories; the option
came in with #27753.

## Logs & Diagnostics (optional)

The reproducer's output on 2026-10-02:

```
config tensors in DRAM, 0 MiB allocated first: conv1d 4.56 s, 27 new kernel binaries ['conv_bmm_tilize', 'halo_gather', 'pack_untilize', 'reader_conv_activations_padded_with_halo_3x3_weights_v2', 'reader_unary_local_l1_copy_backwards', 'reader_unary_sharded_blocks_interleaved_start_id', 'reader_unary_sharded_metal2', 'reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks', 'reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks', 'writer_unary_sharded', 'writer_unary_sharded_blocks_interleaved_start_id_metal2']
config tensors in DRAM, 0 MiB allocated first: conv1d 0.06 s, 0 new kernel binaries []
config tensors in DRAM, 1 MiB allocated first: conv1d 1.49 s, 4 new kernel binaries ['halo_gather', 'reader_conv_activations_padded_with_halo_3x3_weights_v2']
config tensors in L1, 0 MiB allocated first: conv1d 1.52 s, 8 new kernel binaries ['halo_gather', 'reader_conv_activations_padded_with_halo_3x3_weights_v2', 'reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks', 'reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks']
config tensors in L1, 1 MiB allocated first: conv1d 0.07 s, 0 new kernel binaries []
```

## Priority (optional)

P3 (a workaround exists; the cost is start-up time, not correctness). The user's call: P2 if start-up time
matters to the team.

## Impact (optional)

Observed in the CosyVoice2 bring-up (`models/experimental/cosyvoice2`, #56651).
- **Why DRAM:** its HiFT vocoder runs about 80 conv1d per utterance length, all with config tensors in DRAM. In L1
  they accumulated in L1_SMALL, one set per geometry; CosyVoice1 hit the same problem.
- **Measured (2026-09-27, N150):** three fresh processes each built the model and then ran the same fixed two-call
  warm-up.
  - The first compiled 2,373 binaries in 566.5 s.
  - The identical second process compiled **0**, in 46.1 s.
  - A third, identical except for a 1 MiB tensor allocated first, recompiled **1,132** binaries, all `halo_gather`
    plus the two conv reader kernels, in 464.6 s.
- **The workaround:** a fixed warm-up sequence at start-up, so every process allocates identically. It takes
  3.1 min with the kernels on disk and 31.4 min on an empty cache.
- **Prewarm schemes:** kernel-prewarm schemes that replay captured compile recipes (for example the open #55465)
  would inherit the problem, since a recipe captured in one process carries that process's addresses.
