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
