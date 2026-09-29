# CosyVoice2 bring-up — STATUS (2026-09-29)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`. HEAD is `7bd094cc3e`, the last
  commit that reached GitHub before the 09-28 pod expired.
- **Lost with that pod (never pushed):**
  - 8 PR commits: the #36487 candidate (R1), the seam gate (R2), and streaming stages A–C (R3–R5);
  - 4 notes commits: the streaming design, the Stage 3 flow-caching design, the refreshed PR description, the
    redrafted #36487 comment, and FINDINGS/DECISIONS updates.
- **The rebuild spec is `REBUILD_2026-09-29.md`** (this branch). Every number in it is a claim until it is
  re-measured. Two of its numbers conflict with the record (B22).
- **HEAD re-verified on the new pod** (B21):
  - the card is healthy;
  - the device suite passed (204 passed, 3 skipped);
  - token accuracy (95.94 %), the seam gate and TT's generated tokens reproduce 09-28 exactly;
  - the perf test passes (worst RTF 0.634).
- **This pod's environment differs** (RUNBOOK §1–2):
  - KMD 2.9.0, the driver both dead pods ran (O2);
  - no `python_env`: everything runs in `/opt/venv`, plus `inflect`;
  - the checkpoint revision is now recorded and pinned (D35).

## Order of work (user, 09-29; D33)

1. This notes commit.
2. The README.
3. The Stage 1 baseline on HEAD.
4. R1, with R2 overlapping on CPU.
5. The streaming design note.
6. R3.
7. R4, with a hang check first (D36).
8. R5, then stop and report the streaming numbers.

R6 and Stage 3 come after that report.

## Open questions for the user

- None yet this session. The two rebuild-spec conflicts are settled by D37.
