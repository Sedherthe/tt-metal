# CosyVoice2 bring-up — STATUS (2026-09-30, end of session)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings, O = older open
items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **Stopped after the Euler step sweep, as asked (09-30).** The session did the user's 09-30 list in order:
  - D40 kept (D44);
  - the last-0.4 s criterion reset to 15 dB from 36 final chunks (B34);
  - the protected branches pushed once, fast-forward, as authorized;
  - the two drafts written;
  - the firmware restart recorded (B33);
  - the Stage 3 plan, steps 1–3 (B35, B36, B37).
- **PR #56651** (`bringup/cosyvoice2-istft`), `models/experimental/cosyvoice2/`.
  - The local HEAD is `33aa3601eb`, and all of it is on the fork's `backup/2026-09-29-pr`.
  - GitHub's `bringup/cosyvoice2-istft` is at `d2a2439e05`, fast-forwarded from the backup on 09-30 under the
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

- **Notes commits:** `9db66f07d0`, `2f0b029596`, `b7106ce85d`, `d0f5cb94fe`, `50857575e6`, `7c90cc8e67`,
  `8401e5b6b0`, `3b111b1688`, `cb60112fbe`, `ef284f8e25`, `75e2ca04fd`, `51b0719cca`, and the one carrying this file.
  - All are on `backup/2026-09-29-notes`.
  - GitHub's `notes/cosyvoice2` is at `3b111b1688` (the same one-time push); the user pushes it.

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

## Stage 3, measured (09-30)

- **RTF above 1.0 (B35):** only 121-127105-0015. Its tokens spill into a third chunk: 13 tokens that pay a full
  non-streaming flow for 0.52 s of audio. A chunk's flow costs at least 0.82 s, and the flows alone take 0.47–0.74 of
  every utterance's duration.
- **One CFM Euler step (B36):** host-bound at the first chunk's size.
  - 64.7 ms eager, of which the host spends 62.6 ms enqueueing 1,158 ops; 49.2 ms traced.
  - The device's 47.8 ms of kernel time includes 18.5 ms (39 %) of attention head merging.
- **The Euler step count (B37):**
  - WER 0.68 % and SIM within 0.2, at 10, 8, 6 and 5 steps, TT and reference.
  - The audio moves as much on upstream as on TT: at 5 steps, 1.6–1.9 times the port's own distance from upstream.
  - At 5 steps, first audio 0.98–1.14 s and worst streaming RTF 0.82–0.84, so neither target is met.
  - The wavs are in `~/listening/steps/`.

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
  - the noise draws (`cosyvoice2_draws`) and the step sweep (`cosyvoice2_steps`);
  - two kernel caches R6 used (`kc_r6_startup`, `kc_r6_first`);
  - the device profiler's raw logs (`cosyvoice2_runs/0930/cfm_tracy`, 3.2 GB).

  They go when the pod goes; the notes keep the logs.

## Open questions for the user

1. **Pushing:** `bringup/cosyvoice2-istft` (to `33aa3601eb`) and `notes/cosyvoice2`, from the backup branches (D34).
2. **D41, the last 0.1 s** (offered, B34): a criterion there at 12 dB would separate the fix from the old padding by
   10 dB each way. The 0.4 s window cannot separate them on two utterances; the 20 ms level check does.
3. **The step count (B37).** WER and SIM are flat to 5 steps, but the audio moves, as much on upstream as on TT. Keep
   upstream's 10, or listen (`~/listening/steps/`) and choose? Either way it is a maintainers' question too (the
   draft for #54104 asks it).
4. **The CFM's next lever (B36):**
   - the head merge in one op (`nlp_concat_heads`), 18.5 ms of the step's device time now;
   - a traced CFM step in streaming, which needs a design for two live traces (the LLM's and the CFM's);
   - or memory configs, as the plan had it.
5. **Drafts, not posted:**
   - the #54104 streaming update and the #56651 description (`drafts/2026-09-30_*`, refreshed with B35–B37);
   - the #36487 comment (`drafts/2026-09-29_comment_36487_prepare_conv_weights_dram_slicing.md`);
   - the kernel-cache issue (`drafts/2026-09-27_ttnn_issue_conv_dram_config_kernel_hash.md`).

## Next

The Stage 3 plan's steps 1–3 are done (`REBUILD_2026-09-29.md`, "After the rebuild"). What follows depends on the
answers to 3 and 4: memory configs (the plan's step 4), per-bucket CFM traces (its step 5, now shown to matter at
the first chunk's size), or the head merge.
