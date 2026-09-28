# Draft comment for tenstorrent/tt-metal#36487 (not posted)

Status: draft, for the user to post (or not). Written 2026-09-28.

---

We're seeing what looks like the same bug with `ttnn.conv1d`, in CosyVoice2's HiFT vocoder (#54104). Prepared
weights and bias (`ttnn.prepare_conv_weights` / `ttnn.prepare_conv_bias`) give wrong outputs; the raw weight gives
correct ones. It happens whenever the conv runs DRAM-sliced.

Your reproducer also reproduces on our build: with prepare, PCC 0.00035 (values up to 6.4e36); without, PCC 0.999912.

**Environment:** Wormhole N150, KMD 2.3.0, firmware 19.11.0, tt-metal `1f29f312fa`.

**Our geometries.** Inside the real model (real weights, real inputs; the first three are the HiFT vocoder, the
last the flow's CFM estimator), relative L2 error against a float64 torch conv:

| conv | input length | prepared | prepared, given the conv's compute config | raw weight |
|---|---|---|---|---|
| `Conv1d(128->128, k=11, s=1, d=1/3/5, pad 5/15/25)` | 5,120 | 1.2–1.5 | 3e28–3e29 | 0.0036–0.0040 |
| `Conv1d(18->256, k=30, s=15, pad 7)` | 76,801 … 245,761 | 0.92 (7.7 on a silent input) | the same | 0.003 |
| `Conv1d(18->128, k=6, s=3, pad 1)` | 107,521 … 245,761 | 0.69 (0.14–0.19 on a silent input) | the same | 0.003 |
| `Conv1d(320->256, k=3, s=1)` (the flow's CFM estimator, bf16) | 5,120 | 2.13 | — | 0.0026 |

- Input: fp32, TILE, DRAM interleaved (so conv1d auto-slices it through DRAM). Weights bf16.
- Compute: HiFi4, `fp32_dest_acc_en=True`, `packer_l1_acc=True`.
- `Conv1dConfig(weights_dtype=bf16, config_tensors_in_dram=True)`.

**Standalone reproducer** (attached `repro_prepare_conv1d.py`: random weights and inputs, seeded; the same dtypes and
configs). Each arm passes the same conv config and slice config to `prepare_conv_weights`, `prepare_conv_bias` and
`conv1d`:

| conv, input length | DRAM, auto slicing | DRAM, 2 slices | DRAM, 8 slices | `act_block_h_override=1024` | L1 input (no slicing) |
|---|---|---|---|---|---|
| `128->128, k=11`, 5,120 | 3.70 | 1.37 | 1.36 | 1.34 | CBs exceed L1 (all arms) |
| `128->128, k=11, d=3`, 5,120 | **inf** | 1.37 | 1.36 | 1.34 | CBs exceed L1 (all arms) |
| `18->256, k=30, s=15`, 76,801 | 1.23 | 1.23 | 1.23 | 1.25 | CB clash (all arms) |
| `18->128, k=6, s=3`, 107,521 | 1.15 | 1.15 | 1.15 | 1.27 | CB clash (all arms) |
| `18->256, k=30, s=15`, 61,441 | 0.0019 (correct) | **1.23** | **1.25** | 1.25 | 0.0019 (correct) |
| `128->128, k=11`, 10,240 | 0.004 (correct) | CBs exceed L1 | **1.43** | 1.37 | CBs exceed L1 |

The raw weight is correct in every arm that runs (0.002–0.004).

What this suggests:
1. **Without slicing, prepared weights are correct.** `18->256, k=30, s=15` at 61,441 with the input in L1: 0.0019,
   the same as the raw weight.
2. **With explicit DRAM width slicing, prepared weights are wrong at every geometry we tried.** That includes
   geometries where auto slicing happens to give correct results. So it looks like `prepare_conv_weights` lays out
   (or blocks) the weights for a different per-slice configuration than the sliced conv runs.
3. **Auto slicing is right at some lengths and wrong at others.** That would explain #55545's "band of input lengths"
   on the same `Conv1d(128->128, k=11)`. We think #55545 is this bug too.
4. **`act_block_h_override=1024`, #35852's workaround, doesn't help here.**
5. **Giving `prepare_conv_weights` the conv's own compute config (HiFi4 + fp32 accumulation) can make it much
   worse:** 1e29 to inf for the k=11 case.

**Our workaround** (the same as #55545's): on each new geometry, run the fast path once against the raw weight and
keep the prepared weight only where they agree. That caught 20 geometries in our vocoder, and one in the flow once
its largest length bucket grew.

Related: #35852 (DRAM slice config with prepared weights, PCC around 0.5) and #55545 (conv1d, prepared weights wrong
at a band of lengths on Wormhole).
