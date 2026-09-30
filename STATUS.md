# CosyVoice2 bring-up — STATUS (2026-09-30, end of session)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **Stopped after R6, as asked (09-30).** The lost work is rebuilt (R1–R5) and R6 is done. B28's streaming
  WER regression is found and fixed: HiFT's padded calls are masked (B29). Every gate was re-run on the fixed code
  (B31).
- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`.
  - The local HEAD is `d2a2439e05`, and all of it is on the fork's `backup/2026-09-29-pr`.
  - GitHub's `bringup/cosyvoice2-istft` is still at `7bd094cc3e`. The user pushes it (D34).
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

- **Notes commits:** `9db66f07d0`, `2f0b029596`, `b7106ce85d`, `d0f5cb94fe`, `50857575e6`, `7c90cc8e67`,
  `8401e5b6b0`, and the one carrying this file. All are on `backup/2026-09-29-notes`; the user pushes
  `notes/cosyvoice2`.

## The figures (masked HiFT, 09-30)

| target | measured | status |
|---|---|---|
| RTF < 1.0, non-streaming | worst 0.654 (demo) / 0.675 (perf test), aggregate 0.483 / 0.481 | met, enforced |
| token accuracy > 95 % | 95.94 %; the LLM is unchanged | met, enforced |
| WER < 5 % | 0.68 % in each of five noise draws; the reference also 0.68 % in each | met |
| similarity > 0.60 | 95.88 (95.84–95.92) over the draws; the reference 95.22 | met |
| first packet < 500 ms | 1.31–1.50 s (two demo runs); perf test worst 1,469 ms | missed, held in band |
| streaming RTF < 0.4 | worst 1.10–1.12, aggregate 0.84–0.85; perf test worst 1.110 | missed, held in band |

- **Streaming quality:** WER 0.68 % in every draw, similarity 95.83 (95.81–95.87); upstream's streaming 0.68 % and
  95.89 (B30).
- **Start-up:** 3.1 min warm and 31.4 min cold for the buckets, plus 2.5 and 13.0 min for the streaming set (B32).
- **The cold first request:** RTF 66.1 on an empty kernel cache, 2.04 on one already holding its binaries (B32).

## This pod

- **Host and card:**
  - host `app-5ddf2d9d-deployment-5545b7c7f8-c4vpz`;
  - card n150 L at `0000:01:00.0` (`/dev/tenstorrent/2`), KMD 2.9.0, firmware 19.11.0.0.
- **Health:** every 09-30 job ran without a fault. At 09:15 the card was healthy: DRAM OK, heartbeat 116,472
  (`scripts/2026-09-30/tt_smi_end.json`).
  - That heartbeat is below 09-29's 158,131, so the card's firmware restarted overnight. The host did not reboot
    (uptime 92 days), and no reset was issued here.
  - KMD 2.9.0 powers the card down after 5 s idle (`power_policy=Y`), which would explain it. Not confirmed (O2).
- **Environments:** `/opt/venv` for the device side (no `python_env`; `inflect` added), `~/cosyvoice2_ref_env` for
  the reference side (locked, D35). See RUNBOOK §1–2.
- **Data:** under `/home/user/data`:
  - the inputs, the references and the runs (`cosyvoice2_runs/0929`, `0930`);
  - the noise draws (`cosyvoice2_draws`);
  - two kernel caches R6 used (`kc_r6_startup`, `kc_r6_first`).

  They go when the pod goes; the notes keep the logs.

## Open questions for the user

1. **Pushing:** `bringup/cosyvoice2-istft` and `notes/cosyvoice2`, from the backup branches (D34).
2. **D40, the streaming guard** (built on 09-29 before an answer): keep it, or revert to documenting the
   restriction? Not answered yet.
3. **D41's additions** (Claude): the stage A end also gates the last 0.4 s's difference, 20 dB below the signal
   with no floor, and keeps the final chunk's PCC before those 0.4 s. On a quiet ending, PCC measures the port's own
   floor.
4. **The #36487 comment** (`drafts/2026-09-29_comment_36487_prepare_conv_weights_dram_slicing.md`) is drafted, not
   posted.

## Next: the Stage 3 plan (`REBUILD_2026-09-29.md`, "After the rebuild")

1. Per-chunk timings of the utterances with streaming RTF above 1.
2. A profile of one CFM step.
3. An Euler-step sweep with WER/SIM over noise draws (D43's protocol).
4. Memory configs.
5. Per-bucket CFM traces, only if dispatch dominates.
