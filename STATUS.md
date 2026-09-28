# CosyVoice2 bring-up — STATUS (2026-09-28, early morning)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`.
- **Stage 1 is met on the N150:** every target has a `Meets()` verdict recorded in `tests/perf/gates.py` (D18).

  | target | measured |
  |---|---|
  | RTF < 1.0, distinct utterances, warmed buckets | worst 0.633, aggregate 0.481 (B16) |
  | token accuracy > 95 % | 96.37 %, teacher-forced (B12) |
  | WER < 5 % | 0.68 %, the reference also 0.68 % (B11) |
  | SIM > 0.60 | 95.88 (x 100), the reference 95.21 (B11) |

- **Local commits since the user's last push** (each verified, no Co-Authored-By):

  | commit | what |
  |---|---|
  | `b9f75371bc` | non-streaming flow bucketing with a real padding mask (B10) |
  | `b05bc66e8d` | bucketed pipeline warmed at start-up; HiFT cap 1,024 tokens with `SegmentTooLong` (B9, B15, D24) |
  | `1333db5af5` | teacher-forced token accuracy; fp32-logit LLM head (B12, D23) |
  | `c12840947f` | VALIDATION: Stage 1 under the protocol, WER/SIM, the L1 option; `gates.py` verdicts (B16) |

- **Device suite** (the working tree of `1333db5af5`): 201 passed and 3 skipped, in 25:46. The skips are the opt-in
  tracker and the two tests that only run in the reference venv. The perf test ran separately and passed (B16).
- **Start-up** (B15): 76 min on an empty kernel cache, 9.6 min with the kernels on disk. Any code change costs one
  full recompile.
- **Listening pairs** for the HiFT tail are in `~/listening` (B11).

## Stopped here: after the Stage 1 measurement (the user's instruction)

## Proposals (not built)

### P1 — Persist the conv safety checks' verdicts (D17's item d)
- **Why:** they rerun in every process, 182 s of the 577 s warm start-up. They are deterministic: two processes gave
  the same 43 verdicts, geometry for geometry (B15).
- **Keep them, don't drop them.** 20 of the 43 caught real corruption of the prepared-weight path (#55545's class).
- **What:** one JSON file per key set under `~/.cache/cosyvoice2_ttnn/conv_verdicts/`:
  - the key: the conv's place in the model, its parameters, the input length, the dtypes and compute configs;
  - the key also carries the tt-metal commit, the package commit, the checkpoint's hash, the architecture and the
    firmware;
  - the value: which of the three candidates won.
  - On a hit, the pipeline skips the reference conv, the host copies and the float64 arbitration. A raw-weight
    verdict also skips `prepare_conv_weights`.
- **Cost:** skipping the reference convs changes the DRAM allocation sequence (their programs keep config tensors
  alive). So after a code change the first process compiles and writes the verdicts, the second compiles again,
  and the third is warm. That is two cold starts per change instead of one.
- **Saves:** about 182 s of the 577 s warm start-up.

### P2 — Chunked HiFT for non-streaming (D26; Stage 2 needs the same machinery)
- **Upstream's streaming HiFT path** (`CosyVoice2Model.token2wav`, stream=True):
  - each chunk after the first takes the previous chunk's last 8 mel frames (`mel_cache_len=8`);
  - its source's first 3,840 samples come from the previous chunk (`cache_source`), which keeps the sine phase
    continuous;
  - its first 3,840 output samples are crossfaded, Hamming over 7,680, with the previous chunk's held-back tail.
- **The proposal:** run every non-streaming mel through that path in chunks of C frames. C = 512 is a candidate: it
  is the largest bucket where the fast path verified clean (the corruption starts at 640, B15).
  - The last chunk anchors at the end, overlapping the previous one more, so nothing is padded.
  - A mel shorter than C keeps one small bucket.
- **What it buys:**
  - HiFT drops from 12 geometries to 1–2. HiFT is 443 of the 577 s warm start-up and 3,469 of the 4,561 s cold one.
  - There is no length cap (D24), and the tail padding goes away for everything longer than a chunk.
- **Costs and risks:**
  - The output then equals upstream's streaming HiFT for that chunking, not upstream's single-pass output. The seams
    need a gate against single-pass torch HiFT, plus WER/SIM.
  - `TtHiFTStreamingState` (lost, O3) has to be rebuilt; Stage 2 needs it anyway.

## Next (the user's order after this stop)

1. The user's call on P1 and P2.
2. **Streaming** (R12, D22).

## Open questions for the user

- D23: I adopted the fp32-logit head before the cold compile (96.37 % vs 90.66 %, +0.3 ms/token). Keep it?
- P1 and P2 above.
