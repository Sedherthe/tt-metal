# CosyVoice2 bring-up — STATUS (2026-09-27, night)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **PR #56651** (`bringup/cosyvoice2-istft`) lives at `models/experimental/cosyvoice2/` (D19, `958d45152a`).
- **Local commits since the user's last push** (each verified, no Co-Authored-By):

  | commit | what |
  |---|---|
  | `126a7cca79` | `scripts/eval_wer_sim.py`; first WER/SIM scores in `docs/VALIDATION.md` (B7) |
  | `fcb2fd6110` | reference venv on upstream's transformers 4.51.3; both transformers shims removed; advisories dispositioned (D20, B1, B8) |
  | `90ad6140fc` | `docs/VALIDATION.md`: the cross-process kernel-cache misses explained (B3) |

- **Device suite after the move:** 194 passed, 1 skipped (the opt-in tracker), and 1 deselected, in 9 min 18 s on
  the N150. The deselected test is `test_pipeline_perf.py`, which stops at "no verdict recorded" by design until
  D18.
- **Evaluation (B7):**
  - Corpus WER 0.68 % for TT and for the reference, the same single error.
  - SIM 94.90 (TT) vs 95.21 (reference), WavLM-base-plus-sv x 100.
- **The transformers pin (D20)** works, with bit-identical reference output. It trades the two shims for 18
  transformers CVEs, none reachable (B8).
- **B3 is solved in principle.**
  - The cause: DRAM config-tensor addresses sit in the conv and halo kernels' compile-time args.
  - The workaround: a deterministic warm-up at process start. An identical second process compiled 0 kernels.
  - The upstream issue is drafted, not filed.

## Stopped here for the user's review of the B3 results (the user's item 4)

## Next (user's order)

1. **Bucketing (D17, items a–d)**, with its warm-up built on B3's deterministic sequence:
   - a fixed bucket order at process start, before any request allocates;
   - config tensors in L1 is the fallback if the ordering proves fragile, since buckets bound its L1_SMALL use.
2. **The Stage 1 protocol (D18):**
   - distinct utterances with warmed buckets;
   - the start-up warm-up cost and the cold first-request time;
   - then record the `gates.py` verdict.
3. **Streaming** (R12, D22).

## Open questions for the user

- B8: keep the transformers pin (18 CVEs dispositioned), or go back to 5.12.1 plus the two provably exact shims?
- Whether to file the drafted TTNN issue.
