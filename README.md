# CosyVoice2 bring-up notes (not part of the PR)

These are working notes for the CosyVoice2 TTNN bring-up: bounty tenstorrent/tt-metal#54104, draft PR #56651
(branch `bringup/cosyvoice2-istft`). This is an orphan branch that is never merged. Push it after every session.

Read these first:
- `STATUS.md`: one page, rewritten each session. Current state, order of work, next actions.
- `FINDINGS.md`: the numbered registry of findings and bugs (R = the 09-27 review, B = build findings,
  O = older open items, X = fixed).
- `DECISIONS.md`: settled calls, with the reason for each and the condition for revisiting it.
- `RUNBOOK.md`: new-pod checks, environment, run commands, safety rules.

Also here:
- `drafts/`: unposted texts (issue comments, the PR description).
- `design/`: design proposals. `2026-09-27_pipeline_nonstreaming.md` (rev 2) is approved and built through step 5.
- `reviews/`: checks of documents we were handed (`2026-09-27_cosyvoice1_lessons_verification.md`).
- `scripts/2026-09-27/`: one-off checks and their logs behind today's FINDINGS entries (not part of the PR).
- `scripts/perf_2026_09_21/` … `perf_2026_09_25/`, `scripts/vocoder_debug_2026_09_20/`: the dated one-off scripts that
  used to live in the PR tree under `models/demos/audio/cosyvoice2/scripts/`. They moved here unchanged on 09-27
  (R15), so their imports and absolute paths still name that old location. Docstrings in the PR cite them as
  `notes/cosyvoice2:scripts/...`.
- `history/`: the old status handoffs, unchanged. `BRINGUP_STATUS_25_sept.md` (includes the 09-27 additions and
  the 09-18→09-24 history) and `BRINGUP_STATUS_22_sept.md` (the copy committed on the PR branch, due to be
  removed from there).
