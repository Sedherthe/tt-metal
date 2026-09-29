# CosyVoice2 bring-up — STATUS (2026-09-29, end of session)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **Stopped after R5, as planned (D33).** R6 and the Stage 3 plan come next.
- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`.
  - The local HEAD is `6a2ab97dde`, and all of it is on the fork's `backup/2026-09-29-pr`.
  - GitHub's `bringup/cosyvoice2-istft` is still at `7bd094cc3e`. The user pushes it (D34).
- **This session's PR commits**, in order:

  | commit | what | record |
  |---|---|---|
  | `e441e8ad6a` | the README rewritten for reviewers | D33 |
  | `eaadc540b3` | Stage 1 re-verified on this pod | B21 |
  | `5ef75876e5` | the checkpoint revision pinned on both sides; the reference venv locked | D35 |
  | `79349deacd` | R1: #36487, a ROW_MAJOR-prepared weight as the conv checks' fourth candidate | B22, B23 |
  | `c2c387db2f` | R2: the seam gate on nine voiced seams | B24, D39 |
  | `082fad43d6` | R3: streaming stage A, offline from fixed tokens | B25, D38 |
  | `fa4eca1213` | R4: streaming stage B, interleaved with the LLM | B26, D36 |
  | `8ca78aa5a1` | R5: streaming measured; streaming refused without its warm-up | B27, D40 |
  | `6a2ab97dde` | R5: what the 1,259 buffers are (corrects `8ca78aa5a1`'s wording) | B27 |

- **Notes commits:** `9db66f07d0`, `2f0b029596`, `b7106ce85d`, `d0f5cb94fe`, and the one carrying this file. All are
  on the fork's `backup/2026-09-29-notes`. The user pushes `notes/cosyvoice2`.
- **The lost work is rebuilt** (R1–R5 of `REBUILD_2026-09-29.md`), each piece re-measured rather than copied.
  - Didn't reproduce: #36487's reproducer (0.000768, not 0.225; B22) and the "5 of 9 seams agree" caveat (B24).
  - Not re-measured: the spec's cold streaming request (172 s to first audio, RTF 65). That path now raises
    (B27, D40).

## The streaming numbers (R5; B27)

| | run 1 | run 2 | target |
|---|---|---|---|
| time to first audio | 1.365–1.455 s | 1.336–1.479 s | < 0.5 s: missed |
| streaming RTF, per utterance | 0.806–1.057 | 0.787–1.122 | < 0.4: missed |
| streaming RTF, aggregate | 0.853 | 0.843 | |

- **The first chunk:**
  - 0.37–0.47 s until it starts;
  - flow 0.81–0.92 s, of which the CFM takes 0.67–0.73 s;
  - HiFT 0.12 s.

  With a free flow, first audio would be at 0.51–0.59 s.
- **Quality:** WER/SIM 1.36 % / 95.85; upstream's streaming of the same tokens gets 0.68 % / 95.90.
- **Recorded, not enforced:** both targets are `Misses()` in `tests/perf/gates.py`, and no device test enforces
  them yet.

## This pod

- **Host and card:**
  - host `app-5ddf2d9d-deployment-5545b7c7f8-c4vpz`;
  - card n150 L at `0000:01:00.0` (`/dev/tenstorrent/2`), KMD 2.9.0, firmware 19.11.0.0.
- **Health:** healthy after every run. The last check was at 15:05 (heartbeat 158,131;
  `scripts/2026-09-29/r5_tt_smi.json`). No drop this session (O2).
- **Environments:** `/opt/venv` for the device side (no `python_env`; `inflect` added), `~/cosyvoice2_ref_env` for
  the reference side (locked, D35). See RUNBOOK §1–2.
- **Data:** under `/home/user/data`: the inputs, the references, and the runs in `cosyvoice2_runs/0929`. They
  go when the pod goes; the notes keep the logs that back each finding.

## Open questions for the user

1. **D40, the streaming guard.** The 09-29 report left it to the user; it was built before an answer came. Keep
   it, or revert to documenting the restriction?
2. **Pushing:** `bringup/cosyvoice2-istft` and `notes/cosyvoice2`, from the backup branches (D34).
3. **The #36487 comment** (`drafts/2026-09-29_comment_36487_prepare_conv_weights_dram_slicing.md`) is drafted, not
   posted.

## Next: R6, then the Stage 3 plan

- **R6, the numbers D37 left open:**
  - cold start-up (an empty kernel-cache directory) and warm start-up, re-measured (PERF.md's 30.5 and 3.2 min;
    B22);
  - the non-streaming cold first request on chunked HiFT (the spec's RTF 32.5, unverified; B22). It still runs,
    as `demo.py --warmup none` without `--stream` (D40);
  - a cold streaming start: the two warm-ups on an empty kernel cache;
  - then PERF.md's cold figures. Its streaming section is in `8ca78aa5a1`, the README was rewritten in
    `e441e8ad6a`, and P1 is closed as won't-do (D27).
- **The Stage 3 plan** (`REBUILD_2026-09-29.md`, "After the rebuild"):
  1. per-chunk timings of the utterances with RTF above 1;
  2. a profile of one CFM step;
  3. an Euler-step sweep with WER/SIM;
  4. memory configs;
  5. per-bucket CFM traces, only if dispatch dominates.
