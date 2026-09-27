# CosyVoice2 bring-up — findings registry

One entry per finding. Update the **status** in place; don't append narrative. The evidence behind each entry is
in `history/` (dated sections) or in the cited code or commit.

Status values:
- `open`: not started
- `planned`: scheduled in STATUS "Order of work"
- `fixed <hash>`: fixed, with the commit
- `decided`: settled; see DECISIONS.md

Each entry also says whether it is **confirmed** (checked) or a **suspicion** (needs a check).

## R: the 09-27 end-to-end review (numbering kept; the user's adjustments noted)

### R1 — The bounty assignment is at risk (confirmed). Status: open
- Our last comment on #54104 was 09-09. The takeover request came on 09-13, and nobody replied as of 09-27.
- Bounty terms: an issue can be reassigned after "over two (2) weeks" unresponsive, and the PR must be
  submitted while still assigned.
- The draft PR #56651 description is stale.
- Action: push, post the progress comment, and apply the PR description. Drafts are in `drafts/`.

### R2 — No streaming pipeline, which is the core deliverable (confirmed). Status: planned (order steps 2–3)
- There is no pipeline or text-to-speech entry point in `tt/`.
- The frontend is `scripts/vocoder_debug_2026_09_20/cv2_frontend.py`, pulled in with `sys.path` edits.
- `qwen2lm.generate` (`qwen2lm.py:564`) returns the whole list at the end.
- `flow.inference` is non-streaming only.
- `TtHiFTStreamingState` was lost (O3).

### R3 — Stage 1 quality rests on n=1 (confirmed). Status: adjusted
- Evidence:
  - WER 4.17% and SIM 0.889 come from one prompt/target pair (`REF_IDX=0, TGT_IDX=3`) in
    `hf-internal-testing/librispeech_asr_dummy`, validation split. It is not test-clean.
  - Four custom sentences: WER 0/0/0/2.78%, SIM 0.56/0.84/0.79/0.87.
  - Token accuracy is 20 greedy tokens with no speech prompt (`stage1_eval_v4_noisefix.py:118-157`).
  - The tooling differs from the paper's: Whisper `base.en`, and SIM via `campplus.onnx`, which is the model's
    own conditioning encoder.
- **Adjustment:** don't build the 50-utterance paper-matched evaluation yet. Ask the maintainers (question 3 in the
  progress comment) and check what the CosyVoice1 PR reported. Token accuracy moves to the R8 method.

### R4 — Stage 3 is out of reach at 10 Euler steps (confirmed, estimated from measured parts). Status: open
- Streaming RTF from the CFM plus LLM decode alone is 0.46–0.58 with every trace pre-captured, and 0.55–0.69 with
  a capture per chunk. That covers 2–4 s prompts and 5–30 s utterances.
- The first chunk is 274 ms of LLM decode plus 395–635 ms of CFM, before prefill, encoder and HiFT. Buckets under
  384 are extrapolated.
- CosyVoice1 (#52540) was accepted with measured, partly-met Stage 3 targets. Ask the maintainers (question 1).

### R5 — The bucketed streaming encoder can't run the final chunk (confirmed). Status: planned (quick fix)
- `tt/flow/encoder.py:858-861` asserts `valid_length % 25 == 0`, and the mask has no padding term.
- Upstream's last call is `finalize=True` in streaming mode with an arbitrary length (`CosyVoice2Model.tts`).
- Fix: add a padding term (mirror `decoder.py`), plus a test at a non-aligned length with a chunk-only negative
  control.

### R6 — The non-streaming flow may be exposed to SDPA bug #57608 (suspicion). Status: planned (early)
- Fused SDPA is on by default (`decoder.py:654`), non-streaming passes `attn_mask=None` at arbitrary T
  (`decoder.py:756`), and nothing zero-fills tile padding.
- CosyVoice1 saw PCC ≈ 0 at T ≡ 1 mod 32 when `conv1d` left garbage in the tile padding. None of our tested
  lengths are ≡ 1 mod 32.
- Action: sweep fused SDPA against the explicit chain at those lengths, then add
  `ttnn.fill_implicit_tile_padding(k/v, 0)` (CosyVoice1 measured a 1.8% cost).

### R7 — `generate` has no context-limit guard (confirmed). Status: planned (quick fix)
- Nothing checks `pos` against `max_seq_len`. The scripts use 512 with `max_tokens` = 20 × text length.
- Possibly the same problem as R8 and R13; see the recorded check in STATUS.

### R8 — The LLM diverges from torch in zero-shot use (confirmed). Status: open
- 09-23: TT produced more trailing silence than torch in 3 of 4 clips (+1.58 s, +0.22 s, +0.52 s).
- **Method (adjusted):** teacher-forced token accuracy over full sequences with the speech prompt.

### R9 — The HiFT "0.977 gap" is almost certainly F0 phase drift (confirmed from history). Status: adjusted
- 09-20: with device F0, waveform PCC is about 0.11 at 464 frames; with torch F0 injected it is 0.999, and the
  difference is inaudible.
- **Adjusted plan:**
  - No noise-draw check.
  - One confirming run: the 16-frame case with torch F0 injected should reach about 0.999.
  - Verify streaming with torch F0 injected.
  - Judge own-F0 output with spectral metrics.

### R10 — The traced CFM gives about nothing in streaming at capacity 1 (confirmed). Status: decided (D5)
- The simulation gives 0 hits in 45 calls.
- A capture solve costs about the same as an eager warm solve: 697.7 vs 675.3 ms (09-22).

### R11 — The encoder round-trips to the host in every layer (confirmed). Status: planned (optimize step)
- `_rel_shift` (`encoder.py:511-516`) sends a [B, H, T, 2T−1] matrix to torch and back in all 10 layers, and this
  is what blocks tracing the encoder.
- CosyVoice1 already does it on device: `~/reference-cosyvoice1/.../tt/flow/encoder.py:334`.

### R12 — No trace schedule for interleaving LLM decode and flow (confirmed). Status: planned (streaming step)
- The decode trace is scoped to one `generate()`. A trace kept alive across flow and vocoder hung the card (09-21).
- Simplest option: release the LLM trace at each chunk boundary.

### R13 — The upstream text frontend is missing (confirmed). Status: planned (pipeline step)
- Upstream `inference_zero_shot` normalizes text and splits it into segments of at most 80 tokens, calling
  `tts()` once per segment (`cosyvoice.py:93`, `frontend.py:157`).

### R14 — The defaults differ from the reported configuration (confirmed). Status: decided (D14)
- The CFM trace (`decoder.py:639`) and the LLM decode trace are off by default.
- The "warm" RTF is the same request repeated (see the STATUS checks).

### R15 — The PR payload isn't reviewable yet (confirmed). Status: adjusted, continuous
- 11,044 lines of scripts against 6,641 lines of model code.
- 41 of 57 scripts hardcode `/home/user`, and 16 still show `/opt/venv` or `timeout -s KILL` run lines.
- `BRINGUP_STATUS_22_sept.md` is committed, and the README is stale.
- **Adjustment:**
  - Move the dated one-off scripts to this notes branch; don't delete them.
  - Remove `BRINGUP_STATUS_22_sept.md` from the PR branch (a copy is in `history/`).
  - Check whether tt-metal squash-merges before rewriting any commit titles.

### R16 — Over-engineering (confirmed). Status: open (low priority)
- The CFM trace cache keeps an LRU with capacity forced to 1.
- The DRAM-threshold eviction never fires at its default.

### R17 — Rebase risk (suspicion). Status: planned
- The branch forked from the 09-09 `main`; `TtQwen2LM` uses tt_transformers internals.
- **Adjustment:** a timeboxed trial on a side branch, not on the PR branch.

### R18 — `models/experimental` vs `models/demos/audio` (suspicion). Status: open
- Ask the maintainers (question 2).

### R19 — Coverage misses length-specific bugs (confirmed). Status: open
- The worst bugs so far were length-specific (X1, X2, R6).
- 15 of 19 test files use random weights, and there is no real-weight end-to-end pytest.

### R20 — The non-streaming flow golden is our own reimplementation (confirmed gap, low risk). Status: open
- Run upstream's own flow class once, the way the HiFT isolation test did.

## O: older open items

- **O1 — HiFT dtype crash.** Status: planned (quick fix). A bf16 generator (F0 predictor and source) with an fp32
  decoder fails with `TT_FATAL` in `TtStft`'s concat. Fix it at the boundary where the generator hands `s` over.
- **O2 — PCIe drops on two pods.** Status: open (infra). Both ran KMD 2.9.0. The pod with KMD 2.3.0 was stable
  throughout 09-27. Root cause unknown.
- **O3 — `TtHiFTStreamingState` lost on 09-24.** Status: planned (streaming step). Rebuild it per the design in
  `history/BRINGUP_STATUS_25_sept.md` (mel cache 8, source cache 3840, host crossfade).
- **O4 — The torch-CPU LLM reference isn't reproducible across processes** (119 vs 105 tokens with the same seed).
  Status: open (minor).
- **O5 — The cumsum precision test only covers 250 mel frames.** Status: open (nice to have). Add 464 or more.

## X: fixed

| ID | What | Fix | Pinned by |
|---|---|---|---|
| X1 | `TtStft` silently wrong at ≥65,536 samples (prepared-weight conv) | `1cefdec6fc` | `test_stft.py` (4,800 / 64,000 / 65,536 / 222,720) |
| X2 | Silent conv errors, compute-config and weight-prep axes (#55545): per-geometry verification with float64 arbitration | `b35b3d2780`, `1cefdec6fc`, `cbbf3ff708` | `test_conv1d_verification.py` |
| X3 | HiFT ran with zero excitation noise; now draws real noise | `ee1411ea97` | `test_hift_generator_inference.py` (shared noise) |
| X4 | Qwen2LM stop-token and `min_tokens` semantics | `26f17e13a7` / `c44cb8131a` | `test_qwen2lm_generate.py` |
| X5 | CFM trace-reuse corruption from pre-built temb/dt device tensors | `bf64224912` | `test_device_cfm_trace_cache_across_utterances_and_replays` |
| X6 | WER scoring expanded possessive `'s` | `bf64224912` (scripts) | none |
| X7 | DRAM growth per geometry (GeometryWeightCache, `release_caches`) | `51c368a8de` | `test_geometry_cache.py` |
| X8 | Four CFM trace-safety bugs: dt copy compiled after capture, `z_dev` lifetime, fallback freeing tensors it reads, clone readout | `f6fdbfcf93` | the opt-in tracker test (child 6/6) |
| X9 | Tracker test could SIGKILL a device-holding child; now opt-in, no timeout | `f6fdbfcf93`, `08888a6eb0` | none |
| X10 | Streaming CFM uses masked fused SDPA (2.2–7.1× faster than the explicit chain) | `97f702e0fd` | `test_flow_decoder.py` streaming tests |
| X11 | The real-checkpoint check couldn't detect a missing padding term; now a max\|diff\| gate plus a negative control | `5f2aadfef4` | `cfm_streaming_real_checkpoint_check.py` verdict |
