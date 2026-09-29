# CosyVoice2 bring-up — decisions

Settled calls. Change an entry only when its "revisit if" condition happens, and record the change here with
its date.

## Design

### D1 — Bucketed padding plus masks, not a fixed window (09-23)
- **Why:** it keeps upstream's receptive field (whole growing prefix, unlimited left context) and bounds the
  number of distinct device geometries (kernel compiles, conv weight prep). The buckets are linear, step 64
  tokens: at most 28% padding waste and about 9 buckets per 30 s utterance.
- **Revisit if:** padding waste shows up in measured RTF.

### D2 — Masks ported verbatim from upstream (09-23)
- `subsequent_chunk_mask`, `add_optional_chunk_mask`, `mask_to_bias`.
- Goldens compare chunk-causal against chunk-causal, never against full attention (the 09-23 v1 lesson).
- Bucketed and exact-length streaming solves must match to max|diff| ≤ 0.05, and a chunk-only negative control
  must fail that check (X11).

### D3 — Reset the streaming hop to 25 for every utterance (09-25)
- A deliberate deviation. Upstream's `CosyVoice2Model.tts` never resets `token_hop_len` (25→50→100, then it stays
  at 100), which would make time-to-first-packet about 1 s of LLM decode for every later utterance.
- With text splitting (R13), this means once per segment.

### D4 — Euler steps held at 10 (09-22)
- **Revisit** for Stage 3 (R4): WER, SIM and listening at 8 and 6 steps first, not 5.

### D5 — CFM trace capacity fixed at 1 (09-25); streaming CFM runs eager (09-27)
- A lazy multi-slot trace cache is not allocation-safe.
- At capacity 1, streaming gets about zero hits, and a capture costs about the same as eager (R10).
- **Revisit** only with a design that pre-captures every bucket.

### D6 — HiFT runs in fp32 with real weights (09-18)
- In bf16, PCC collapses to about 0.49 (`conv_post`'s values before `exp`).

### D7 — Keep the conv resolver's per-geometry verification (09-20)
- It is justified by real silent errors (X1, X2).
- It costs time at first sight of each geometry, so geometries must be pre-warmed before measuring
  time-to-first-packet.

### D8 — HiFT audio crossfade on the host (09-24)
- The tensors are tiny (3,840 samples). This matches upstream `fade_in_out`.

### D9 — Allocation-tracker test is opt-in and runs in its own pytest invocation (09-27)
- `ttnn.close_device` keeps UMD's `CHIP_IN_USE` lock until process exit, so a child process can't open the card.
- The test has no timeout and never sends SIGKILL.

### D17 — Non-streaming bucketing before streaming, with gates (user, 09-27 evening)
- **Why:** first sight of each exact length dominates distinct-utterance RTF (B2). A finite bucket set can be
  pre-warmed, and streaming needs the same machinery.
- **Flow.** A real padding mask goes through attention: the non-streaming path the partial-mask guard refuses today.
  - Gate: bucketed vs exact-length output.
  - Negative control: the same comparison without the mask must fail.
- **HiFT.** Pad the mel with a silence-level value, not zeros, and trim the audio. Measure the tail error with torch
  F0 injected (D16).
- **The bucket set** derives from the 80-token segment cap. Report how long warming every bucket takes.
- **Check** how much of the "kernels on disk" RTF (2.0–2.9) is the conv safety checks re-running in each process.
  If it is most of it, propose persisting their results to disk, keyed by geometry and commit.
- **Outcome (09-28):** built (`b9f75371bc`, `b05bc66e8d`). The sets: 15 flow buckets, 12 HiFT, 8 LLM prefill
  lengths. The conv caches never evict in bucketed mode. Config tensors stay in DRAM with the deterministic warm-up,
  because the L1 option doesn't fit (B15). The checks rerun per process (182 s of 577 s), and persisting their
  verdicts is proposed in STATUS.

### D18 — Stage 1 RTF protocol (user, 09-27 evening)
- **The measurement:** distinct utterances with warmed buckets.
- **Also reported:** the start-up warm-up cost and the cold first-request time.
- **The `gates.py` verdict** for `rtf_nonstreaming` is recorded only after bucketing (D17).
- **Outcome (09-28):** `Meets()` recorded for rtf_nonstreaming (worst 0.633), token_accuracy (96.37 %), wer
  (0.68 %) and speaker_similarity (95.88). The start-up cost and the cold first request are in B15 and B16.

### D23 — fp32-logit LLM head (Claude, 09-28; the user can reverse it)
- Adopted before the cold warm-up, so the bucket set compiled once instead of twice. The alternative was measuring
  Stage 1, then changing the head, then recompiling everything and measuring again.
- Measured: 96.37 % vs 90.66 % token accuracy, for 0.3 ms per decode step.
- One field reverses it: `CosyVoice2Config.llm_head_logits_dtype="bfloat16"`.

- **Outcome (user, 09-28, second round):** keep the fp32-logit head as the default. VALIDATION.md explains the
  noise-floor reasoning (`9ccd53edd0`).

### D24 — HiFT cap: 2,048 frames; past it, a clear error (user, 09-28)
- `max_segment_speech_tokens=1024`. Past the cap, `SegmentTooLong` names the segment's length, and a test covers it.
- Chunked HiFT is the eventual fix (D26).

### D25 — Waits: sentinel files, never process state (user, 09-28)
- Each job writes its exit code to a sentinel file as its last action. The chain waits on the file, with a timeout
  that reports rather than kills.

### D26 — Chunked HiFT for non-streaming: propose after Stage 1, don't build (user, 09-28)
- Fixed-size mel chunks with upstream's streaming cache and crossfade. That leaves one or two HiFT geometries, no
  cap, and no tail padding. It is Stage 2 work anyway. The proposal is in STATUS.
- **Outcome (user, 09-28, second round):** approved as the next build. It is built and gated (`da90d8cc84`, B18).
  The cap it removed (D24) is back to upstream's 1,600.

### D27 — P1 (persist the conv check verdicts): deferred until after chunked HiFT (user, 09-28)
- **Outcome (user, 09-28, after chunked HiFT): won't do.**
  - The checks are 21.5 s of the 195 s warm start (PERF.md).
  - The note recording this was lost with the 09-28 pod. It is recorded here on 09-29 from the rebuild spec.

### D28 — Token accuracy on more sequences, with the sample size next to the number (user, 09-28)
- Done: B19 (`c7df6d00d5`).

### D29 — The 20 corrupted geometries against #36487 (user, 09-28)
- If it is the same bug, draft a comment for #36487 (don't post it); if different, draft a separate issue. It is
  the same bug (B17); the comment is in `drafts/`.

### D30 — PERF.md records start-up plainly (user, 09-28)
- Done in `9ccd53edd0`; updated for chunked HiFT in the Stage 1 re-verification commit.

### D31 — Streaming final chunk: upstream parity, the flow runs non-streaming (user, 09-28; recorded 09-29)
- The final chunk runs the existing bucketed non-streaming flow over all tokens, which is already warmed. Only the
  new frames are emitted.
- **Why:** it is what upstream does.
  - `CosyVoice2Model.tts` passes no `stream` to its last `token2wav` (`cosyvoice/cli/model.py:367-373` at
    `074ca6dc9e80`).
  - So `stream` defaults to False (`:292`), and the flow gets `streaming=False` (`:301`).
  - Checked in the source on 09-29.
- The earlier streaming final-chunk path does not match upstream.
- D22's trace release before the final chunk still applies.
- The note recording this was lost with the 09-28 pod. It is recorded here from the rebuild spec.

### D32 — Stage 3 flow caching: dropped (user, 09-28; recorded 09-29)
- **Why** (09-28 measurements, lost with the design note, not re-measured):
  - the prompt is 15–19 % of the flow;
  - the gain would be about 0.1 s of time to first audio;
  - it costs 2.3 MB of device memory per mel frame;
  - it breaks final-chunk parity (D31).
- **Revisit if:** R5's first-chunk breakdown puts the prompt's share of the flow much higher.

### D21 — Evaluation (user, 09-27)
- **ASR:** Whisper large-v3. WER is per utterance and at corpus level.
- **Speaker similarity:** `microsoft/wavlm-base-plus-sv` (`WavLMForXVector`) cosine × 100, with the model named.
  CAM++ is a diagnostic only.
- **The comparison:** TT against the PyTorch reference (the post-fix reference only), never against the paper's
  0.745.
- **First pass:** the existing wavs, no device time.

### D22 — Streaming final chunk (user, 09-27; design change (a))
- **Release the LLM decode trace** before the final chunk runs.
- **Warm every final-chunk bucket.** Don't rely on "needs no warming".
- **Our caveat:** HiFT's final-chunk length is not bucketed today. D17's HiFT bucketing is meant to close that.
- **09-28:** the final chunk's flow runs non-streaming, as upstream's does (D31). Its flow buckets are the
  non-streaming set, already warmed. The rebuild spec pads the final HiFT chunk to 128 or 256 frames.

## Process

### D10 — Device runs (09-27)
- No `timeout` wrapper.
- If a run looks stuck, tell the user.
- Stop only with SIGINT, never SIGKILL: it can wedge the card.

### D11 — Git (standing)
- No commit or push unless the user asks in that turn.
- No `Co-Authored-By`.
- The user pushes. Report every verified commit immediately.

### D12 — Order of work, breadth first (user, 09-27)
1. Correctness quick fixes: R5, R6, R7, O1, and the HiFT torch-F0 confirmation.
2. A non-streaming `tt/pipeline.py`, with text normalization and splitting and an explicit configuration.
3. The streaming loop: R12, eager streaming CFM, HiFT streaming.
4. Measure time-to-first-packet and RTF, and evaluate per the maintainers' answer.
5. Optimize: R11, then the step count.

Alongside steps 1–2, a timeboxed rebase trial on a side branch. Cleanup continuously.

### D13 — Adjustments to the 09-27 review (user, 09-27)
- **R9:** no noise-draw check. One confirming run: the 16-frame case with torch F0 injected should reach about
  0.999. Verify streaming with torch F0 injected; judge own-F0 output with spectral metrics.
- **R10/R14:** check how the 0.43–0.52 "warm" RTF was measured. It was the same request repeated (checked
  09-27), so future RTF must come from distinct utterances (D14).
- **R7/R8/R13:** possibly one problem. Check whether the 09-23 clips with extra trailing silence had total
  sequence lengths near `max_seq_len=512`.
- **R3/R8:** measure token accuracy teacher-forced over full sequences with the speech prompt. This replaces the
  20-token figure.
- **R3 (evaluation scope):** don't build the 50-utterance paper-matched evaluation yet. Ask the maintainers, and
  check what the CosyVoice1 PR reported.
- **R17:** a timeboxed rebase trial on a side branch later, not on the PR branch.
- **R15:** move dated one-off scripts to the notes branch rather than deleting them. Remove the committed
  `BRINGUP_STATUS_22_sept.md` from the PR branch. Check whether tt-metal squash-merges PRs before rewriting commit
  titles.
- **R6:** do the SDPA sweep early; it could silently corrupt non-streaming output.

### D14 — Performance reporting (09-27)
- RTF and time-to-first-packet come from distinct utterances, not repeated requests.
- Always state the configuration (traces on or off, steps, prompt length) and whether the kernel cache was warm.
- Estimates never appear in results tables.

### D16 — How HiFT is verified (09-27; the user's R9 plan, confirmed by measurement)
- **Mechanism checks** (source module, decode, iSTFT, and later the streaming caches and splice) inject the torch
  F0 and gate on PCC ≥ 0.99 plus max|diff| over the affected region. With torch F0, TT reaches PCC 0.9995–0.9999.
- **The own-F0 path** is judged with spectral metrics (log-mel or multi-resolution STFT distance), never
  waveform PCC. A few Hz of F0 deviation drives waveform PCC to 0.1–0.4 within 16–208 frames, and the 09-20
  listening test found it inaudible.
- **Revisit** if the F0 predictor's precision improves enough (O6: HiFi3 vs HiFi4) to make own-F0 waveform
  comparisons meaningful.

### D15 — Where the notes live (09-27)
- The orphan branch `notes/cosyvoice2`, never merged into the PR.
- STATUS.md is rewritten each session; findings and decisions are updated in place.
- Dated one-off scripts move here (D13).

### D19 — Location: `models/experimental/cosyvoice2/` (user, 09-27 evening)
- CosyVoice1 moved to `models/experimental/cosyvoice` at the maintainers' request (mtairum, 09-23).
- The move is a `git mv` as its own commit, with paths and tests updated.

### D20 — Reference venv: pin transformers to upstream's version (user, 09-27 evening)
- Pin transformers to the version upstream CosyVoice2's `requirements.txt` specifies. Keep a compatibility patch
  only if the pin can't work, and document why.
- **First outcome (09-27, `fcb2fd6110`), superseded:**
  - Pinned 4.51.3, with tokenizers 0.21.4 and huggingface-hub 0.36.2.
  - Both transformers shims were removed; `reference_env.py` refuses any other version.
  - The reference output is unchanged bit for bit.
  - Cost: 18 transformers CVEs to disposition (B8).
- **Revised (user, 09-27 night, `027d30ec33`); the state since:**
  - Back to transformers 5.12.1 with the two shims. `reference_env.py` refuses any other version.
  - The shims are exact: 4.51.3 with no shims gave bit-identical reference output on all seven cases.
  - 4.51.3's 18 CVEs (B8) outweighed pinning upstream's own version. `docs/security.md` has the reasoning.
  - Corrected here on 09-29; this entry had kept the first outcome.

### D33 — Rebuild order and scope (user, 09-29)
1. The notes commit.
2. The README. It goes early because the PR is public and the README says only the iSTFT exists. It is CPU work,
   done while the device runs.
3. The Stage 1 baseline on HEAD.
4. R1, with R2 overlapping on CPU.
5. The streaming design note.
6. R3.
7. R4, with a hang check first.
8. R5.
9. R6.
10. Stage 3.

This session stops after R5, with the streaming numbers. R1 must land before R4. It also matters for R3, whose
108-frame HiFT length hit #36487.

### D34 — Git on the rebuild day (user, 09-29; updates D11)
- Local commits are authorized for this session, one per verified chunk.
- Every commit is pushed at once to `backup/2026-09-29-pr` or `backup/2026-09-29-notes`, and its hash reported to
  the user.
- Never push `bringup/cosyvoice2-istft` or `notes/cosyvoice2`. The user pushes those.
- No `Co-Authored-By`.
- The pre-commit hook is installed for the PR repo. The worktrees share it, so notes commits use
  `PRE_COMMIT_ALLOW_NO_CONFIG=1`.

### D35 — Reproducible inputs (user, 09-29)
- **Record the checkpoint revision and pin it** in the prepare/reference scripts, so a re-download can't change
  results. `FunAudioLLM/CosyVoice2-0.5B` is at `eec1ae6c79877dbd9379285cf8789c9e0879293d`.
- **Lock the reference venv:** a constraints file next to `requirements-reference.txt`, so indirect dependencies
  can't drift again. On 09-29 three of them had drifted (`scripts/2026-09-29/README.md`). It is committed with R1
  or on its own.

### D36 — KMD 2.9.0 (user, 09-29; updates D10)
- R4 starts with a hang check. KMD 2.9.0 is the driver both dead pods ran (O2).
- If the card drops at any point:
  - stop;
  - make sure the backup branches hold everything;
  - report the exact error and what was running.
- Don't retry resets in a loop.

### D37 — The two rebuild-spec conflicts (user, 09-29; B22)
- **#36487's reproducer:** R1 re-measures it. Today's number is used everywhere, including the comment draft. The
  discrepancy is noted in FINDINGS.
- **"RTF 32.5 on chunked HiFT":** unverified. R6 measures it fresh, together with cold start-up (an empty
  kernel-cache directory) and warm start-up.

### D38 — Streaming's final HiFT call is padded at its end, and its tail gated on level (Claude, 09-29; the user can reverse it)
- The spec pads the final call to 128 or 256 frames at the end. Its last ~0.4 s then differ from upstream's, as
  bucketing's tail does (B11).
- The alternatives were rejected:
  - running every final length exactly makes an unbounded geometry set;
  - anchoring the final call to the end with earlier frames as context would change the crossfade region instead.
- The gate checks the final chunk's body on PCC, and its last 0.4 s on level: 20 dB below the signal, or under
  −50 dBFS (B25).

### D39 — The seam gate's thresholds on the voiced set (Claude, 09-29; the user can reverse it)
- The relative error over the crossfade is bounded at 0.10, the spec's bound, and the control must fail at every
  seam.
- max |diff| is no longer gated: an absolute bound follows loudness.
- Whole-signal PCC moves from 0.998 to 0.995, set from the first voiced-seam measurement (B24).
