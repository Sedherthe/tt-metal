# CosyVoice2 bring-up notes (not part of the PR)

These are working notes for the CosyVoice2 TTNN bring-up: bounty tenstorrent/tt-metal#54104, draft PR #56651
(branch `bringup/cosyvoice2-istft`). This is an orphan branch that is never merged. Push it after every session.

Read these first:
- `STATUS.md`: one page, rewritten each session. Current state, order of work, next actions.
- `FINDINGS.md`: the numbered registry of findings and bugs (R = the 09-27 review, O = older open items,
  X = fixed).
- `DECISIONS.md`: settled calls, with the reason for each and the condition for revisiting it.
- `RUNBOOK.md`: new-pod checks, environment, run commands, safety rules.

Also here:
- `drafts/`: unposted texts (issue comments, the PR description).
- `history/`: the old status handoffs, unchanged. `BRINGUP_STATUS_25_sept.md` (includes the 09-27 additions and
  the 09-18→09-24 history) and `BRINGUP_STATUS_22_sept.md` (the copy committed on the PR branch, due to be
  removed from there).
