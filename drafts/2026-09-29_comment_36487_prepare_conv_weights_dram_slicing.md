# Draft comment for tenstorrent/tt-metal#36487 (not posted)

Status: draft, for the user to post (or not). Redrafted 2026-09-29 on a second board, with every number re-measured
that day (D37). It replaces `2026-09-28_comment_36487_prepare_conv_weights_dram_slicing.md`: that draft's reproducer
figure (0.00035) and its table were measured on the 09-28 board, and the ROW_MAJOR finding is new.

---

We're seeing what looks like the same bug with `ttnn.conv1d`, in CosyVoice2's vocoder and flow decoder (#54104).
Weights and bias from `ttnn.prepare_conv_weights` / `ttnn.prepare_conv_bias` give wrong outputs wherever the conv
slices its input through DRAM, while the raw weight gives correct ones.

**Declaring `input_layout=ttnn.ROW_MAJOR_LAYOUT` to prepare fixes those geometries**, although the activation is
TILE. It also breaks the geometries where the TILE declaration was right.

**Your reproducer on our build** (Wormhole N150, KMD 2.9.0, firmware 19.11.0, tt-metal `1f29f312fa`):

| weights and bias | PCC vs torch |
|---|---|
| prepared with `input_layout=TILE_LAYOUT` (as written) | 0.000768 (the output holds inf) |
| prepared with `input_layout=ROW_MAJOR_LAYOUT` | 0.999912 |
| raw, not prepared | 0.999912 |

**`ttnn.conv1d`: relative L2 error against a float64 torch conv.**
- Random weights and inputs, seeded.
- Activations TILE, DRAM interleaved; fp32 for the first six rows, bf16 for the last.
- bf16 weights; HiFi4 with fp32 accumulation; `config_tensors_in_dram=True`.
- Prepare and conv get the same conv config.

| conv | input length | prepared, TILE declared | prepared, ROW_MAJOR declared | raw weight |
|---|---|---|---|---|
| `Conv1d(128->128, k=11, d=1/3/5)` | 4,320, 5,120, 8,320 | 1.36–3.78 | 0.0039–0.0055 | the same as ROW_MAJOR |
| `Conv1d(128->128, k=11)` | 10,240 | 0.0040 | 1.37 | 0.0040 |
| `Conv1d(18->256, k=30, s=15)` | 76,801 … 245,761 (8 lengths) | 1.23 | 0.0040 | the same |
| `Conv1d(18->256, k=30, s=15)` | 12,961 … 61,441 (5 lengths) | 0.0019 | 1.13–1.17 | 0.0019 |
| `Conv1d(18->128, k=6, s=3)` | 107,521 … 245,761 (6 lengths) | 1.15 | 0.0039 | the same |
| `Conv1d(18->128, k=6, s=3)` | 12,961 … 61,441 (5 lengths) | 0.0018 | 1.07 | 0.0018 |
| `Conv1d(320->256, k=3, pad (2, 0))`, batch 2 | 5,120 | 1.26 | 1.36 | 0.0036 |

- Over the 36 geometries behind this table, ROW_MAJOR is right wherever TILE is wrong (23 cases), and wrong wherever
  TILE is right (11 cases).
- In all of them the correct declaration gives exactly the raw weight's error, to the last digit.
- The exceptions: the flow's `Conv1d(256->256, k=3)` at the same length is right all three ways (not in the table),
  and the last row is wrong both ways. We don't have an explanation for the last row.
- Our script: [attach `r1_prepare_layout.py`].

**Where we think it comes from**, reading the code at our commit (please correct us):
- `get_input_channels_alignment` (`conv2d_utils.cpp:92-99`) returns a smaller channel alignment for a ROW_MAJOR input
  *or a sliced op*, and `TILE_WIDTH` otherwise.
- `ttnn.conv1d` routes DRAM inputs through DRAM width slicing by default (`conv1d.cpp:82-88`).
- `prepare_conv_weights` never takes the DRAM path for a 1-D conv. At `prepare_conv2d_weights.cpp:1313`,
  `is_dram_conv = ... && !is_conv1d` ("Conv1D doesn't support DRAM").
- So for a TILE input it aligns channels to 32 (`:1441-1442`), while the sliced op uses the smaller alignment.
  Declaring ROW_MAJOR gives prepare the sliced op's alignment. Where the op doesn't slice, TILE is the right
  declaration.
- If that's right, prepare would need to make conv1d's slicing decision itself.
- Your conv2d reproducer shows the same ROW_MAJOR effect, though prepare does take the DRAM branch there. So the
  alignment may not be the whole story.

**Our workaround:**
- Every new geometry runs once against a raw-weight reference.
- On a disagreement, a float64 host conv chooses among the TILE-prepared weight, a ROW_MAJOR-prepared weight and the
  raw weight, keeping a prepared weight whenever one is right.
- **In our non-streaming pipeline** (17 flow lengths, 2 vocoder lengths), one geometry is left corrupted: the
  flow's `Conv1d(320->256, k=3)` at 5,120. Real weights give 2.126 declaring TILE and 2.267 declaring ROW_MAJOR,
  so it runs on the raw weight (0.0026).
- **Streaming** runs the vocoder at 108 and 208 frames, where every k=11 resblock conv is TILE-wrong. There the
  ROW_MAJOR candidate keeps them prepared.
- **Some convs reject a ROW_MAJOR-prepared weight outright:** a width-sharded `Conv1d(80->512, k=3)` fails
  `act_matrix_width == weight_matrix_height` at validation.

Related: #35852 (DRAM slice config with prepared weights, PCC around 0.5) and #55545 (conv1d, prepared weights wrong
at a band of lengths on Wormhole). The band in #55545 would be the lengths at which conv1d slices.
