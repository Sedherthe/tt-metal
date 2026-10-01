# CosyVoice2 bring-up — STATUS (2026-09-30, end of session)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## 2026-10-01, in progress (this block is replaced by the end-of-session rewrite)

- **The 09-30 pod ended overnight.** Nothing in git was lost: everything below is on `backup/2026-09-29-pr`
  (`213afe7909`) and `backup/2026-09-29-notes`. Backups continue on those two branches (the user, 10-01).
- **The new pod** (B42): n150 L at `0000:e1:00.0`, healthy. The environment is rebuilt, and the reference venv
  equals the lock. The reference side is being regenerated; then the backup tip gets re-verified on this card
  (`scripts/2026-10-01/`).
- **Found on GitHub, not in these notes** (R1): #56651 is ready for review, not a draft, and mtairum **approved**
  it on 09-30 on `33aa3601eb` after Tier-3 CI. The backup's three later commits would move the head past it. Pushing
  is the user's call (D34).

## Where things stand (09-30, end of session)

- **Stopped after lever (b)'s proposal, as asked (09-30, D45).** This round did, in order:
  - the protected branches pushed once, fast-forward, confirmed by the user;
  - the stage A end gate's last-0.1 s check at 12 dB, and all three end checks shown both ways (B38);
  - the "Generated with Claude Code" line removed from the PR draft. No attribution lines anywhere (the user's
    rule, kept in memory);
  - the Euler step count as a config option, default 10, with the sweep in PERF.md (B39);
  - lever (a): the CFM's heads merged by `nlp_concat_heads` (B40);
  - lever (b): the traced CFM during streaming, proposed with a tracker proof and measured in a prototype (B41).
- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`.
  - The local HEAD is `213afe7909`, and all of it is on the fork's `backup/2026-09-29-pr`.
  - GitHub's `bringup/cosyvoice2-istft` is at `33aa3601eb`, fast-forwarded from the backup on 09-30 under the
    user's one-time authorization. The user pushes it (D34).
- **The PR commits of 09-29 and 09-30**, in order:

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
  | `6a2ab97dde` | R5: what the 1,259 buffers are | B27 |
  | `ed1c3ad1c5` | HiFT's padded calls masked; D41's end gate; the "you" clip's regression test | B28, B29, D41, D42 |
  | `5317572d0c` | WER/SIM over five noise draws, TT and reference, Stage 1 and streaming | B30, D43 |
  | `ba56920c77` | the Stage 1 and streaming gates re-run on the masked HiFT | B31 |
  | `d2a2439e05` | R6: the streaming figures enforced by a device test; start-up and the cold first request | B32 |
  | `f899ad51a2` | stage A's last-0.4 s criterion at 15 dB, from 36 final chunks | B34 |
  | `3ab7d78bf0` | Stage 3 step 1: which utterance streams above RTF 1.0, and why (docs) | B35 |
  | `7be3b2aa48` | Stage 3 step 2: one CFM Euler step profiled on the device (docs) | B36 |
  | `33aa3601eb` | Stage 3 step 3: the Euler step sweep, 10/8/6/5 (docs) | B37 |
  | `ea95ce1d66` | stage A's end gate adds the last 0.1 s at 12 dB | B38 |
  | `1dcdc10059` | the Euler step count as a config option (`flow_n_timesteps`, default 10) | B39 |
  | `213afe7909` | the CFM's heads merged by `nlp_concat_heads`, bit-identical | B40 |

- **Notes commits:**
  - `9db66f07d0`, `2f0b029596`, `b7106ce85d`, `d0f5cb94fe`, `50857575e6`, `7c90cc8e67`, `8401e5b6b0`, `3b111b1688`,
    `cb60112fbe`, `ef284f8e25`, `75e2ca04fd`, `51b0719cca`, `045d09dc43`, `f6a63fef2e`, `531f703808`,
    `a5c0782036`, and the one carrying this file.
  - All are on `backup/2026-09-29-notes`.
  - GitHub's `notes/cosyvoice2` is at `045d09dc43` (this round's one-time push); the user pushes it.

## The figures (masked HiFT, 09-30)

| target | measured | status |
|---|---|---|
| RTF < 1.0, non-streaming | worst 0.654 (demo) / 0.675 (perf test), aggregate 0.483 / 0.481 | met, enforced |
| token accuracy > 95 % | 95.94 %; the LLM is unchanged | met, enforced |
| WER < 5 % | 0.68 % in each of five noise draws; the reference also 0.68 % in each | met |
| similarity > 0.60 | 95.88 (95.84–95.92) over the draws; the reference 95.22 | met |
| first packet < 500 ms | 1.31–1.50 s (two demo runs); perf test worst 1,469 ms | missed, held in band |
| streaming RTF < 0.4 | worst 1.10–1.12, aggregate 0.84–0.85; perf test worst 1.110 | missed, held in band |

- **These headline runs predate the head merge (B40).** On the five-draw protocol, the merge moved:
  - Stage 1 RTF: worst 0.648–0.667 → 0.620–0.664, aggregate 0.477–0.484 → 0.458–0.470;
  - streaming RTF: worst 1.082–1.128 → 1.061–1.086.

  The output is bit-identical.
- **Streaming quality:** WER 0.68 % in every draw, similarity 95.83 (95.81–95.87); upstream's streaming 0.68 % and
  95.89 (B30).
- **Start-up:** 3.1 min warm and 31.4 min cold for the buckets, plus 2.5 and 13.0 min for the streaming set (B32).
- **The cold first request:** RTF 66.1 on an empty kernel cache, 2.04 on one already holding its binaries (B32).

## Stage 3, measured (09-30)

- **RTF above 1.0 (B35):** only 121-127105-0015, a 13-token third chunk paying a full non-streaming flow. The flows
  alone take 0.47–0.74 of every utterance.
- **One CFM Euler step (B36, B40):**
  - host-bound at the first chunk's size: 62.4 ms eager, of which ~61 ms is the host enqueueing 1,102 ops;
  - 31.3 ms traced;
  - device kernel time 30.0 ms, 47.8 before the merge.
- **The Euler step count (B37):** WER and SIM are flat at 10, 8, 6 and 5 steps, TT and reference, but the audio moves
  as much on upstream as on TT. At 5 steps first audio is 0.98–1.14 s. The default stays 10, as a config option
  (B39, D45).
- **The traced CFM during streaming, a prototype (B41):**
  - 16 of 17 buckets traceable; the 5,120-frame one has a raw-weight conv verdict (#36487);
  - under the tracker, all traces alive with no failure;
  - trace region 123.6 MB;
  - first audio 1.337–1.443 → 0.955–1.046 s, streaming RTF aggregate 0.805 → 0.725;
  - audio within log-mel L1 0.067–0.100 of eager.

## This pod

- **Host and card:**
  - host `app-5ddf2d9d-deployment-5545b7c7f8-c4vpz`;
  - card n150 L at `0000:01:00.0` (`/dev/tenstorrent/2`), KMD 2.9.0, firmware 19.11.0.0.
- **Health:** every 09-30 device job ran without a device fault. At 09:46 the card was healthy: DRAM OK, heartbeat
  120,243 (`scripts/2026-09-30/tt_smi_0946.json`).
  - All seven cards on the host restarted their firmware together at ~16:55 on 09-29, while this card was idle
    (B33). The host did not reboot. The cause is unknown (a driver reload or a reset of every card).
- **Environments:** `/opt/venv` for the device side (no `python_env`; `inflect` added), `~/cosyvoice2_ref_env` for
  the reference side (locked, D35). See RUNBOOK §1–2.
- **Data:** under `/home/user/data`:
  - the inputs, the references and the runs (`cosyvoice2_runs/0929`, `0930`);
  - the noise draws (`cosyvoice2_draws`), the step sweep (`cosyvoice2_steps`) and the merge's draws
    (`cosyvoice2_merge`);
  - two kernel caches R6 used (`kc_r6_startup`, `kc_r6_first`);
  - the device profiler's raw logs (`cosyvoice2_runs/0930/cfm_tracy`, `merge_tracy`, 3.2 GB each).

  They go when the pod goes; the notes keep the logs.

## Open questions for the user

1. **Pushing:** `bringup/cosyvoice2-istft` (to `213afe7909`) and `notes/cosyvoice2`, from the backup branches (D34).
2. **The step count:** listen (`~/listening/steps/`) and choose; the default stays 10 until then (D45).
3. **Lever (b), the traced CFM:** build it? Before building:
   - WER/SIM over five draws for the traced configuration;
   - whether to trace the final chunk too (17 more traces);
   - the pipeline's trace region raised to at least 124 MB.
4. **Drafts, not posted:**
   - the #54104 streaming update and the #56651 description (`drafts/2026-09-30_*`; they predate B38–B41);
   - the #36487 comment;
   - the kernel-cache issue.

## Next

Lever (c) in the user's order (D45): re-profile before any memory-config work, then build (b) if approved.
