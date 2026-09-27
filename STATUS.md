# CosyVoice2 bring-up — STATUS (2026-09-27)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md (R* = the 09-27 review, O* = older open items, X* = fixed) or DECISIONS.md (D*). Older narrative
lives in `history/`.

## Where things stand

- **Bounty #54104:** assigned to Sedherthe on 08-24. Our last comment on the issue was 09-09. RosMengHeang asked
  to take it over on 09-13, and no maintainer had replied as of 09-27. Bounty terms allow reassignment after
  "over two (2) weeks" unresponsive (R1).
- **Draft PR #56651:** head `Sedherthe:bringup/cosyvoice2-istft`. The fork is at `5c65be445c`. Four commits
  are local and unpushed: `f6fdbfcf93`, `97f702e0fd`, `5f2aadfef4`, `08888a6eb0`. The PR description is stale;
  a refreshed draft is in `drafts/`, and so is the progress-comment draft.
- **Tests:** 160 collected; 159 passed plus 1 opt-in skipped, at `97f702e0fd` on the N150 (09-27).
- **Pod (09-27):** `app-5131bc20-deployment-c96966b99-8fcs2`, N150 at `0000:41:00.0`, KMD 2.3.0, firmware
  19.11.0. No PCIe problems all session (O2).

## Bounty map (details in the 09-27 review, FINDINGS R1–R20)

| Requirement | State |
|---|---|
| iSTFT as matmul + overlap-add, checked against `torch.istft` | Verified (`test_istft.py`, 256–65,537 frames) |
| HiFT vocoder | Built; the composed real-weight path is in no test (R19). F0 phase drift explains the waveform-PCC gap (R9) |
| Qwen2 LLM via tt_transformers | Built; weakly verified (R3, R8); no context-limit guard (R7) |
| Flow decoder, chunk-aware | Built. Streaming CFM verified with real weights; streaming encoder fails the final chunk (R5) |
| End to end on N150 | Scripts only; no pipeline module or demo (R2) |
| Stage 1 WER/SIM on a representative set | One utterance plus 4 sentences (R3); scope waits for the maintainers (D13) |
| Non-streaming RTF < 1.0 | 0.43–0.52 on the same request repeated, traces on; distinct-utterance RTF not measured (R14, D14) |
| Stage 2: memory configs; iSTFT overlap-add streaming | Not started (R2, O3) |
| Stage 3: TTFP < 500 ms, streaming RTF < 0.4 | Unmeasurable yet; estimated out of reach at 10 steps (R4) |

## Order of work (agreed 09-27, D12)

1. **Correctness quick fixes:** R5 (encoder padding term for the final chunk); R6 (SDPA sweep at T ≡ 1 mod 32,
   early); R7 (context-limit guard); O1 (HiFT dtype crash); the HiFT torch-F0 confirmation (R9: the 16-frame
   case with torch F0 injected should reach about 0.999).
2. **Non-streaming `tt/pipeline.py`:** frontend moved into `tt/`, text normalization and splitting (R13),
   explicit configuration (R14).
3. **Streaming loop:** trace schedule (R12), streaming CFM run eager (R10, D5), HiFT streaming verified with torch
   F0 injected (R9, O3).
4. **Measure TTFP and RTF on distinct utterances (D14); evaluate as the maintainers answer (D13).**
5. **Optimize:** on-device `rel_shift` (R11), Euler step count validated on WER (D4).

Alongside steps 1–2: a timeboxed rebase trial on a side branch (R17). Cleanup continuously (R15).

## Recorded checks, not yet run

- **R14:** how the "warm" RTF was measured. Read on 09-27: it's the same request repeated in-process
  (`traced_encoder_and_cfm.py:383-432`: new, r1, r2, r3 on the same text with `seed=0`), and the CFM trace is
  released after every repeat. Optimistic, so future RTF comes from distinct utterances (D14).
- **R7/R8/R13:** did the 09-23 clips with extra trailing silence have total sequence lengths near
  `max_seq_len=512`? They may be one problem.
- **R3/R8:** token accuracy measured teacher-forced over full sequences with the speech prompt. This replaces the
  20-token greedy figure.
- **R15:** does tt-metal squash-merge PRs? Check before rewriting any commit titles.
- **R3:** what did the CosyVoice1 PR (#52540: PERF.md, docs/VALIDATION.md) report for WER/SIM?

## Next actions (user)

- Push `bringup/cosyvoice2-istft` (4 commits) and `notes/cosyvoice2`.
- Post the progress comment (`drafts/2026-09-27_issue_54104_progress_comment.md`) after the push; apply the PR
  description (`drafts/2026-09-27_pr_56651_description.md`).
- The untracked `models/demos/audio/cosyvoice2/BRINGUP_STATUS_25_sept.md` in the PR working tree is now a copy of
  `history/BRINGUP_STATUS_25_sept.md`. Delete it once this branch is pushed.
