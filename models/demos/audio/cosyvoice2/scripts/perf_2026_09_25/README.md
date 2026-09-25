# perf_2026_09_25 — streaming CFM round (rebuilt)

Evidence for the streaming (chunk-causal, bucketed) CFM work in `tt/flow/decoder.py`. The 2026-09-23/24 originals
of the first three scripts were lost with an instance before they were committed; these are recreations, re-run
on the N150 on 2026-09-25, not copies of the old numbers.

| script | device | what it answers |
|---|---|---|
| `masked_sdpa_investigation.py` | yes | Does fused SDPA take the padding + chunk-causal mask correctly, and how fast is it compared with the explicit chain? |
| `cfm_streaming_real_checkpoint_check.py` | yes | Real `flow.pt`: bucketed vs. exact-length streaming solve (aligned and non-aligned valid lengths, trace reuse), and the per-bucket cost |
| `cfm_trace_cache_thrashing_simulation.py` | no | Under the real hop schedule, what trace-cache capacity would help, and by how much (measured costs) |
| `nonstreaming_timing_old_vs_new.py` | yes | Is the non-streaming path's timing unchanged against the pre-change decoder (`5c65be445c`)? |
| `../perf_2026_09_23/bucket_sizing_simulation.py` (PART 3) | no | The real growing hop schedule (`real_hop_schedule_lengths`) against `GeometryWeightCache` |

Run the device scripts from a directory other than the repo root, with
`PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal`.

## Results (2026-09-25, N150)

**Masked SDPA, op level** (`[2, 8, T, 64]` bf16, q_chunk 128 / k_chunk 256):

- Building the mask on device as the broadcast sum `chunk[1,1,T,T] + pad[B,1,1,T]` exactly matches the host sum.
- Accuracy vs. float64: PCC 0.9997 at every T and valid length.
- Speed: 2.2x / 4.9x / 7.1x faster than the explicit chain at T = 384 / 768 / 1536. A mask costs 2.1–3.5x over no mask.
- The op's default program config (no q/k chunk sizes) is slower than q128/k256 at T = 768 and 1536. At T = 384 it is slightly faster.
- Leak probe (large values in the padded rows of V): a chunk-only mask collapses to PCC 0.13 / 0.30 / 0.76 at the non-aligned lengths. The full mask stays at 0.9997.
- With random V, the same missing padding term still scores above 0.99, which is why the probe is needed.

**Real checkpoint, 10-step streaming solve** (first run):

- The TT bucketed solve is bit-identical to the TT exact-length solve (PCC 1.0, max|diff| 0) at all six (valid, bucket) pairs: 300/310→384, 650/660→768, 1500/1510→1536.
- Against the fp32 torch chunk-causal reference: 0.9962–0.9988.
- In that run the non-aligned cases did not reuse the aligned case's trace; the script has since been fixed (see its docstring). The fixed version has not been re-run yet: the card became unrecoverable at teardown of the first run (see below).

| bucket (mel) | capture | recapture (warm) | streaming steady | ms/Euler step | non-streaming steady |
|---|---|---|---|---|---|
| 384 | 734.8 | 744.4 | 501.6 | 50.16 | 406.6 |
| 768 | 1069.6 | 1071.9 | 820.4 | 82.04 | 733.4 |
| 1536 | 2071.7 | 2063.9 | 1628.4 | 162.84 | 1377.7 |

All times are ms per 10-step traced solve. Bucket 1536 is over the 1 s per-chunk budget; that is re-measured, not extrapolated.

**Trace-cache thrashing** (5 back-to-back 30 s utterances, measured costs):

- Every CFM call in an utterance lands in a distinct bucket, so a single-slot cache never hits.
- Any capacity below the session's full bucket range gives zero hits.
- A full-range cache would cut total CFM time from 64.5 s to 52.5 s (−19%). Capture overhead is 23% of single-slot time, not the ~100% the lost 09-24 analysis stated.
- A lazy multi-slot cache is not allocation-safe, so `TtCausalConditionalCFM` refuses any capacity other than 1 (see its docstring). This number is a ceiling for a future pre-warmed design, not an available win.

**Real hop schedule** (upstream `CosyVoice2Model.tts`, re-read 2026-09-25):

- The hop goes 25(+prompt pad) → 50 → 100 → 100 …, plus one finalize call. Mid-stream valid lengths are always chunk-aligned; the finalize call's length is arbitrary.
- `tts()` never resets `token_hop_len`, so later utterances on the same model object start at a hop of 100. Both variants are simulated.
- Per-utterance reset: 9 flow calls per 30 s utterance, 80.0% `GeometryWeightCache` hit rate, breaking point 96.8 MB/bucket.

## Not yet run

- `nonstreaming_timing_old_vs_new.py`: written but not run (the card went down first).
- The fixed `cfm_streaming_real_checkpoint_check.py`: needs a re-run.
