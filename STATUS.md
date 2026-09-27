# CosyVoice2 bring-up — STATUS (2026-09-27, evening)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md (R* = the 09-27 review, O* = older open items, X* = fixed) or DECISIONS.md (D*). Older narrative
lives in `history/`.

## Where things stand

- **Bounty #54104:** assigned to Sedherthe. Our last comment on the issue was 09-09. The progress comment is
  drafted (`drafts/`), and the user is posting it. There's a 2-week reassignment rule (R1).
- **PR #56651:** head `bringup/cosyvoice2-istft`. **No longer a draft as of 09-27 08:52 UTC**, but its head is
  still `5c65be445c` and its title is the old one. **8 local commits are unpushed** (all verified on the N150,
  no Co-Authored-By):
  - `f6fdbfcf93` CFM trace-safety fixes
  - `97f702e0fd` streaming CFM
  - `5f2aadfef4` evidence scripts
  - `08888a6eb0` tracker test with no timeout
  - `026325a73d` LLM context budget (R7)
  - `fe1e3dedb8` encoder final chunk (R5)
  - `5e8ac6e433` #57608 regression test (R6)
  - `544d588018` HiFT dtype boundary (O1)
- **Suite at `544d588018`:** 175 passed, 1 skipped (the opt-in tracker) of 176, 4 min 36 s warm (09-27, N150).
- **Notes:** `notes/cosyvoice2` on the fork (`29e5d0806e` plus this update).
- **Pod:** `app-5131bc20-deployment-c96966b99-8fcs2`, N150 at `0000:41:00.0`, KMD 2.3.0. Stable all day (O2).

## Today (09-27): order-of-work step 1, correctness quick fixes

| Item | Result |
|---|---|
| R7, context limit | **Fixed.** `generate()` refuses a call that doesn't fit `max_seq_len`. The 09-23 clips were **not** near 512 (286/350/376/493), so R7 doesn't explain R8. |
| R5, encoder final chunk | **Fixed.** Key-padding term added. With real weights the chunk-only control fails the region gate (0.23–0.39 vs ≤0.067), while its whole-output PCC stayed ≥0.998. |
| R6, SDPA #57608 | **Not reproducible on Wormhole** (op level with verified planted padding, and a real-weight estimator sweep at T ≡ 1 mod 32). No fix; regression test with a harness control. |
| O1, HiFT dtype crash | **Fixed.** A single dtype boundary at `TtHiFTDecoder.decode`; bf16 generator with fp32 decoder reaches PCC 0.99999. |
| R9, HiFT torch F0 | **Confirmed: F0 phase drift, not a bug.** Real `hift.pt`, fp32, a real speech mel. With torch F0 injected, PCC 0.99989 at 16 frames (0.99962 at 108, 0.99953 at 208). With its own F0, PCC 0.40 / 0.35 / 0.115 from a max F0 deviation of 0.5–7.4 Hz. Own-F0 waveform PCC is not a usable metric (D16). |

## Order of work (D12, unchanged)

1. ~~Correctness quick fixes (R5, R6, R7, O1, the R9 confirmation)~~. Done 09-27.
2. **Next: non-streaming `tt/pipeline.py`.** Design proposed in `design/2026-09-27_pipeline_nonstreaming.md`,
   pending review: the frontend boundary, installs, explicit configuration, the demo.
3. The streaming loop (R12, eager streaming CFM, HiFT streaming verified with torch F0 injected).
4. Measure TTFP and RTF on distinct utterances; evaluate as the maintainers answer.
5. Optimize: R11, the step count, and O6 (HiFi3 vs HiFi4).

Alongside: a timeboxed rebase trial on a side branch (R17), and continuous cleanup (R15).

## Decisions needed from the user

- **Installs:** `onnxruntime` / `torchaudio` (CPU) / whisper log-mel (or reimplementations) / `inflect`, or a
  separate frontend venv like CosyVoice1's (design doc). This blocks the R8 re-run and the pipeline's frontend.
- **Push** the 8 commits, then post the comment and apply the PR description.

## Recorded checks, not yet run

- R8: teacher-forced token accuracy over full sequences with the speech prompt; re-run the 09-23 silence clips
  once the frontend deps exist.
- R15: does tt-metal squash-merge PRs? Check before rewriting commit titles.
- R3: what the CosyVoice1 PR reported for WER/SIM.
