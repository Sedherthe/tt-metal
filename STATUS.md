# CosyVoice2 bring-up — STATUS (2026-09-28, morning)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`.
- **Stage 1 is met on the N150 with chunked HiFT:** every target has a `Meets()` verdict in `tests/perf/gates.py`.

  | target | measured |
  |---|---|
  | RTF < 1.0, distinct utterances, warmed buckets | worst 0.628, aggregate 0.479; perf test worst 0.621 (B20) |
  | token accuracy > 95 % | 95.94 % over 5,003 positions, 27 sequences, 4 speakers (B19) |
  | WER < 5 % | 0.68 %, the reference also 0.68 % (B20) |
  | SIM > 0.60 | 95.87 (x 100), the reference 95.21 (B20) |

- **Start-up** (B20): 30.5 min on an empty kernel cache (9,959 kernels), 3.2 min with the kernels on disk. Before
  chunked HiFT these were 76 min and 9.6 min. Any code change costs one cold start.
- **Local commits since the user's last push** (each verified, no Co-Authored-By):

  | commit | what |
  |---|---|
  | `9ccd53edd0` | PERF.md with the start-up figures; VALIDATION on why fp32 logits are the default (D23, D30) |
  | `da90d8cc84` | chunked HiFT, gated against upstream's streaming HiFT; cap back to 1,600; #36487 diagnosis in VALIDATION (B17, B18) |
  | `c7df6d00d5` | token accuracy on 27 sequences, 4 speakers: 95.94 % (B19, D28) |
  | `7bd094cc3e` | Stage 1 re-verified on chunked HiFT (B20) |

- **Device suite** (the chunked-HiFT tree): 204 passed and 3 skipped (the opt-in tracker and the two reference-venv
  tests) in 15:39. The perf test ran separately and passed (B20).
- **#36487:** our 20 (now 21) corrupted conv geometries are that bug (B17). The comment draft is
  `drafts/2026-09-28_comment_36487_prepare_conv_weights_dram_slicing.md`; the user posts it, or not.
- **The earlier B15/P2 bucket error:** the six resblock corruptions are at the 128-frame bucket, not 640. That is
  corrected in VALIDATION (`da90d8cc84`) and in B15.

## Stopped here: chunked HiFT gated and Stage 1 re-verified on it (the user's instruction)

## Next (for the user to order)

1. **Streaming** (R12, D22). Chunked HiFT is its vocoder half: the same cache, with streaming chunk sizes.
2. **P1** (persist the check verdicts, D27): 21.5 s of the 195 s warm start now, and it costs a second cold start
   per code change (see the 09-28 early-morning STATUS in git history).
3. **The flow now dominates the warm start** (157 of 195 s). Its two largest buckets (2,048 and 2,560 tokens) only
   serve prompt + speech above 1,792 tokens.
4. The package README still says only the iSTFT is implemented. It is stale and misleading for reviewers; not
   touched here.

## Open questions for the user

- Post the #36487 comment (B17)?
- The DRAM kernel-hash issue draft (09-27): the user files it after reading into it.
