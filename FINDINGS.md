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

### R2 — No streaming pipeline, which is the core deliverable (confirmed). Status: non-streaming built `0dbe9c44f4`; streaming planned after bucketing (D17)
- There is no pipeline or text-to-speech entry point in `tt/`.
- The frontend is `scripts/vocoder_debug_2026_09_20/cv2_frontend.py`, pulled in with `sys.path` edits.
- `qwen2lm.generate` (`qwen2lm.py:564`) returns the whole list at the end.
- `flow.inference` is non-streaming only.
- `TtHiFTStreamingState` was lost (O3).

### R3 — Stage 1 quality rests on n=1 (confirmed). Status: in progress. There is now a fixed corpus (`scripts/corpus.py`, 2 speakers × 3 targets) and a PyTorch reference run, and D21 sets the scoring
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

### R5 — The bucketed streaming encoder can't run the final chunk (confirmed). Status: **fixed `fe1e3dedb8`** (09-27)
- **09-27:** added `streaming_attn_bias_torch` (upstream's `masks & chunk_masks`) at both encoder stages and
  removed the assert.
  - Tests at non-aligned lengths, with large values in the padded rows. With real weights, over the last partial
    chunk: fixed max|diff| is 0.016–0.067 against TT exact and the torch reference; the chunk-only control is
    0.23–0.39 and fails the gate (PCC ≥ 0.999, max|diff| ≤ 0.125).
  - The control's **whole-output PCC was 0.998–0.9998**, so a whole-output gate would have passed it.
  - Random-init weights can't carry this control: the leak stays at bf16-noise level there.
  - Bucketed and exact-length encoder runs are *not* bit-identical (2–4 bf16 ulps), unlike the CFM.
- `tt/flow/encoder.py:858-861` asserts `valid_length % 25 == 0`, and the mask has no padding term.
- Upstream's last call is `finalize=True` in streaming mode with an arbitrary length (`CosyVoice2Model.tts`).
- Fix: add a padding term (mirror `decoder.py`), plus a test at a non-aligned length with a chunk-only negative
  control.

### R6 — The non-streaming flow may be exposed to SDPA bug #57608 (suspicion). Status: **closed on Wormhole; regression test `5e8ac6e433`** (09-27)
- **09-27:** the only fused-SDPA call site in our code is `decoder.py` `_sdpa`. The LLM uses tt_transformers'
  prefill SDPA with `is_causal` on sequences padded to a multiple of 128, plus the separate decode op.
- Op level: 1e30 planted in K and V padding, by both methods and read back to confirm, leaves the output
  bit-identical under every config, including #57608's own. **#57608 does not reproduce on this N150 build**;
  it was reported only on Blackhole.
- Real-weight estimator sweep at T ≡ 1 mod 32 from 161 to 1537: fused SDPA equals the explicit chain and the
  torch reference (PCC 0.9982–0.9995). No fix.
- The regression test at T=449 checks padding immunity, with a harness control that must change the output.
  **On Blackhole this test would be expected to fail**; the fix there would be `fill_implicit_tile_padding`
  (CosyVoice1 measured a 1.8% cost).
- Fused SDPA is on by default (`decoder.py:654`), non-streaming passes `attn_mask=None` at arbitrary T
  (`decoder.py:756`), and nothing zero-fills tile padding.
- CosyVoice1 saw PCC ≈ 0 at T ≡ 1 mod 32 when `conv1d` left garbage in the tile padding. None of our tested
  lengths are ≡ 1 mod 32.
- Action: sweep fused SDPA against the explicit chain at those lengths, then add
  `ttnn.fill_implicit_tile_padding(k/v, 0)` (CosyVoice1 measured a 1.8% cost).

### R7 — `generate` has no context-limit guard (confirmed). Status: **fixed `026325a73d`** (09-27)
- **09-27:** added `required_max_seq_len(prefix, max_tokens)` and `TtQwen2LM.prefix_len`. `generate()` raises
  `ValueError` before prefill when a call doesn't fit.
- Tests: the arithmetic, and a call sized exactly to the budget passes while one more token is refused. The
  unfixed code does not raise.
- The 09-23 clips' totals were 286 / 350 / 376 / **493** of 512, so the limit was never binding. It can't explain
  R8, but clip 4 was 19 tokens from a silent overflow.
- Nothing checks `pos` against `max_seq_len`. The scripts use 512 with `max_tokens` = 20 × text length.
- Possibly the same problem as R8 and R13; see the recorded check in STATUS.

### R8 — The LLM diverges from torch in zero-shot use (confirmed). Status: open
- **09-27:** it is *not* the context limit (R7). Clip 1, with the largest extra silence (+1.58 s), totals 286
  tokens and could reach at most 337.
- Re-running the 09-23 synthesis needs the frontend: `onnxruntime`, plus `torchaudio` and the whisper log-mel,
  or reimplementations of both. That needs install approval (see the pipeline design doc).
- 09-23: TT produced more trailing silence than torch in 3 of 4 clips (+1.58 s, +0.22 s, +0.52 s).
- **Method (adjusted):** teacher-forced token accuracy over full sequences with the speech prompt.

### R9 — The HiFT "0.977 gap" is F0 phase drift (confirmed 09-27). Status: **closed (not a bug); method in D16**
- **09-27 run** (`scripts/2026-09-27/hift_torch_f0_injection.py`): real `hift.pt`, fp32, a real speech mel
  (LibriSpeech-dummy item 3, the most voiced window), shared `sine_noise`.
  - With torch F0 injected into the NSF source: PCC **0.99989** at 16 frames, 0.99962 at 108, 0.99953 at 208
    (max|diff| 0.026–0.055 on waveforms peaking at 0.39–0.76). The composition is correct.
  - With its own F0: PCC 0.40 / 0.35 / 0.115, from a max F0 deviation of 3.56 / 0.49 / 7.4 Hz. That is lower
    than 09-24's 0.977 because this window is 15/16 voiced; the mechanism is the same.
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

### R13 — The upstream text frontend is missing (confirmed). Status: **fixed `d651a5edfc`** (`tt/text.py`, with parity tests against upstream)
- Upstream `inference_zero_shot` normalizes text and splits it into segments of at most 80 tokens, calling
  `tts()` once per segment (`cosyvoice.py:93`, `frontend.py:157`).

### R14 — The defaults differ from the reported configuration (confirmed). Status: decided (D14). `CosyVoice2Config.reported()` makes it explicit, and the pipeline refuses environment switches (`0dbe9c44f4`)
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

### R19 — Coverage misses length-specific bugs (confirmed). Status: partly addressed. `test_pipeline_api.py` (`0dbe9c44f4`) is a real-weight end-to-end pytest over six lengths
- The worst bugs so far were length-specific (X1, X2, R6).
- 15 of 19 test files use random weights, and there is no real-weight end-to-end pytest.

### R20 — The non-streaming flow golden is our own reimplementation (confirmed gap, low risk). Status: open
- Run upstream's own flow class once, the way the HiFT isolation test did.

## B: build findings (09-27, steps 3–5)

### B1 — Upstream's LLM reference ran broken under transformers 5.x (confirmed). Status: **fixed `0d687d840e`** (shims); D20 re-examines it with the pinned transformers
- **The decode mask.** Upstream's `inference_wrapper` passes a length-1 all-ones mask at each decode step.
  transformers 4.51 dropped it; 5.x right-pads it with zeros (`masking_utils.prepare_padding_mask`), so each step
  attended to position 0 only.
  - Measured (`scripts/2026-09-27/ref_mask_check.py`): max |d log p| of 16.4 against a no-cache forward, and 3.1e-5
    with the mask spanning cache + input.
  - Unshimmed, the first two corpus cases generated until `max_len` (180 and 440 tokens).
- **The load dtype.** 5.x also loads `from_pretrained` in the config's dtype (bf16 for BlankEN) where 4.51 loaded
  fp32. The first matmul failed on mixed dtypes.

### B2 — Distinct-utterance RTF is dominated by first sight of each new length (confirmed). Status: planned (D17)
- **Three regimes, N150, reported configuration** (`docs/VALIDATION.md`, `0dbe9c44f4`):

  | regime | RTF |
  |---|---|
  | new length, kernels not compiled | 21–75 |
  | new length, kernels on disk | 2.0–2.9 |
  | a length the process already ran | 0.39–0.56 |

- **Where the time goes, per regime:**
  - Cold: HiFT 155–370 s and the flow 23–57 s. The LLM is 2–4 s in every regime.
  - Steady state: HiFT 0.11–0.34 s, the flow 0.8–1.5 s, the LLM 0.9–3.2 s.
- **Non-streaming lengths are exact**, so every distinct utterance is a new geometry. The earlier "warm RTF ≈ 0.5"
  was a repeated request (D14).

### B3 — The disk kernel cache is only partly reused across processes (confirmed; mechanism unknown). Status: open
- **The same lengths, same seeds, same tokens.** The demo compiled them, yet the pytest processes recompiled HiFT
  for every one.
- **The flow was reused once:** 494 tokens. Two other lengths recompiled.
- **Identical call sequences did reuse each other's kernels:** two pytest processes running the same sequence.
- **All processes share one cache build key.** A hypothesis, unverified: something process-dependent, such as a
  DRAM address, enters HiFT's conv kernel compile arguments.

### B4 — Device memory across consecutive different-length utterances (confirmed). Status: closed
- **L1_SMALL:** 0 B/bank for all nine calls of `test_pipeline_api.py`.
- **DRAM:** grows 10–22 MiB/bank per new length (prepared conv weights), and stays flat on repeats. The
  free-DRAM eviction bounds it; that eviction was not exercised.

### B5 — The fp32 F0/source default costs nothing in warm RTF (confirmed). Status: closed (O6 follow-up)
- Warm HiFT is equal within 4 ms for fp32 and bf16 (`scripts/2026-09-27/f0_dtype_rtf_check.py`).

### B6 — TT and the PyTorch reference sample different token sequences under the same seed (expected). Status: note
- The torch version is the same; the logits differ.
- Token counts are within 8 % (for example, 176 vs 191).

## O: older open items

- **O1 — HiFT dtype crash.** Status: **fixed `544d588018`** (09-27). `TtHiFTDecoder.decode` converts `mel` and
  `s` to its dtype on entry. Test: bf16 generator with fp32 decoder, PCC 0.999994–0.999998 and max|diff|
  0.0004–0.0012; the unfixed code fails with the concat same-dtype TT_FATAL. The original problem: a bf16
  generator (F0 predictor and source) with an fp32 decoder failed with `TT_FATAL` in `TtStft`'s concat.
- **O2 — PCIe drops on two pods.** Status: open (infra). Both ran KMD 2.9.0. The pod with KMD 2.3.0 was stable
  throughout 09-27. Root cause unknown.
- **O3 — `TtHiFTStreamingState` lost on 09-24.** Status: planned (streaming step). Rebuild it per the design in
  `history/BRINGUP_STATUS_25_sept.md` (mel cache 8, source cache 3840, host crossfade).
- **O4 — The torch-CPU LLM reference isn't reproducible across processes** (119 vs 105 tokens with the same seed).
  Status: open (minor).
- **O5 — The cumsum precision test only covers 250 mel frames.** Status: open (nice to have). Add 464 or more.
- **O6 — HiFi4 + fp32 accumulation on Wormhole (09-27).** Status: **closed, not a lever**
  (`scripts/2026-09-27/f0_fidelity_ab.py`). Real `hift.pt` F0 predictor, three real utterances (931 voiced
  frames), against the fp32 torch F0:
  - HiFi4 vs HiFi3 at fp32: mean |df| 0.228 vs 0.230 Hz, max 8.4 vs 7.2 Hz, phase drift 1.35 vs 1.38 cycles.
    Within noise.
  - **dtype matters instead:** bf16 gives 0.70 Hz mean, 31–33 Hz max, 2–5 voiced/unvoiced flips, 2.2–2.3 cycles
    of drift. The pipeline defaults the F0/source path to fp32.
  - Neither fidelity approaches the ~0.03 Hz needed for sub-0.1-cycle drift (CosyVoice1 PERF.md §7), which
    confirms D16.
  - Untested lever: the F0 classifier's `ttnn.linear` has no compute config.

  The original concern: tt-metal warns: "On Wormhole with fp32
  accumulation, output accuracy can be worse with HiFi4 than HiFi3 due to a hardware bug." Our conv resolver's
  "accurate" config is HiFi4 + fp32 accumulation (`hifigan/conv.py`). Check the F0 predictor's and the
  decoder's accuracy with HiFi3; relevant to R9's F0 drift.

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
| X12 | LLM context budget unchecked (R7) | `026325a73d` | `test_qwen2lm_generate.py::test_device_generate_refuses_request_over_context_budget`, `test_required_max_seq_len` |
| X13 | Streaming encoder's final (non-aligned) chunk: key-padding term (R5) | `fe1e3dedb8` | `test_flow_checkpoint.py::test_device_streaming_encoder_final_chunk_real_checkpoint` (with control), `test_upsample_conformer_encoder.py::..._final_chunk_bucketed_matches_exact` |
| X14 | HiFT dtype boundary, bf16 source path into fp32 decoder (O1) | `544d588018` | `test_hift_generator_inference.py::test_device_hift_generator_bf16_source_fp32_decoder` |
| X15 | #57608 guard, not reproducible on Wormhole (R6) | `5e8ac6e433` (test only) | `test_flow_decoder.py::test_device_decoder_fused_sdpa_ignores_tile_padding_at_t_1_mod_32` |
| X16 | The reference LLM under transformers 5.x: fp32 load, and the decode mask (B1) | `0d687d840e` | `scripts/2026-09-27/ref_mask_check.py` (notes) |
