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

### B1 — Upstream's LLM reference ran broken under transformers 5.x (confirmed). Status: **resolved by the pin, `fcb2fd6110`** (D20). Under upstream's transformers 4.51.3 with no shims, upstream reproduces the shimmed run's tokens and audio bit for bit, 7 of 7 cases. The pin's cost is B8.
- **The decode mask.** Upstream's `inference_wrapper` passes a length-1 all-ones mask at each decode step.
  transformers 4.51 dropped it; 5.x right-pads it with zeros (`masking_utils.prepare_padding_mask`), so each step
  attended to position 0 only.
  - Measured (`scripts/2026-09-27/ref_mask_check.py`): max |d log p| of 16.4 against a no-cache forward, and 3.1e-5
    with the mask spanning cache + input.
  - Unshimmed, the first two corpus cases generated until `max_len` (180 and 440 tokens).
- **The load dtype.** 5.x also loads `from_pretrained` in the config's dtype (bf16 for BlankEN) where 4.51 loaded
  fp32. The first matmul failed on mixed dtypes.

### B2 — Distinct-utterance RTF is dominated by first sight of each new length (confirmed). Status: **fixed by bucketing, `b05bc66e8d`**: RTF 0.43–0.63 on distinct utterances after the warm-up (B16)
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

### B3 — The disk kernel cache is only partly reused across processes (confirmed). Status: **workaround in production (`warmup_buckets()`, `b05bc66e8d`); upstream issue drafted, the user files it**
- **Cause.** With `config_tensors_in_dram=True`, the conv reader kernels and the halo reader kernels take their config
  tensors' DRAM addresses as compile-time args (`conv2d_op_sharded_program_factory.cpp:871`,
  `conv2d_op_width_sharded_program_factory.cpp:562`, `untilize_with_halo_program_factory.cpp:307-316`, unchanged on
  `main`). A binary on disk is reused only when the new process's config tensors land at the same DRAM addresses.
- **Item 4a** (`scripts/2026-09-27/b3_deterministic_warmup.py`): three fresh processes run the same two-call warm-up.

  | process | kernels compiled | process time |
  |---|---|---|
  | 1, cold | 2,373 | 566.5 s |
  | 2, identical | **0** | 46.1 s |
  | 3, identical except 1 MiB allocated first | **1,132**, all `halo_gather` + the two conv reader kernels | 464.6 s |

- **Standalone reproducer** (`repro_conv_dram_config_kernel_hash.py`): a single conv1d. With the config in DRAM,
  a 1 MiB shift recompiles; with the config in L1, it doesn't.
- **Not reported upstream** (searched issues and PRs, 09-27). Draft: `drafts/2026-09-27_ttnn_issue_conv_dram_config_kernel_hash.md`.
- **09-28, in production:** an identical second process compiled 0 of the first's 19,068 binaries, with the same DRAM
  figures after every geometry (B15). Any code change still costs one full recompile.

### B4 — Device memory across consecutive different-length utterances (confirmed). Status: closed
- **L1_SMALL:** 0 B/bank for all nine calls of `test_pipeline_api.py`.
- **DRAM:** grows 10–22 MiB/bank per new length (prepared conv weights), and stays flat on repeats. The
  free-DRAM eviction bounds it; that eviction was not exercised.

### B5 — The fp32 F0/source default costs nothing in warm RTF (confirmed). Status: closed (O6 follow-up)
- Warm HiFT is equal within 4 ms for fp32 and bf16 (`scripts/2026-09-27/f0_dtype_rtf_check.py`).

### B6 — TT and the PyTorch reference sample different token sequences under the same seed (expected). Status: note
- The torch version is the same; the logits differ.
- Token counts are within 8 % (for example, 176 vs 191).

### B7 — WER and speaker similarity: TT indistinguishable from the PyTorch reference (confirmed). Status: measured (`126a7cca79`)
- Six LibriSpeech utterances, 147 words, Whisper large-v3.

  | | corpus WER | mean SIM (WavLM-base-plus-sv x 100) |
  |---|---|---|
  | TT | 0.68 % | 94.90 |
  | reference | 0.68 % | 95.21 |

- The single error is the same substitution in both. Scores are unchanged under the transformers pin.

### B8 — The transformers pin reintroduces advisories (confirmed). Status: dispositioned (`fcb2fd6110`); the user's call
- transformers 4.51.3 carries 18 distinct CVEs (30 OSV records), 3 of them HIGH; 5.12.1 carries none.
- None is on the reference venv's code path. The one on `from_pretrained` (CVE-2026-4372) is ruled out by grepping
  the two pinned configs.
- The alternative, 5.12.1 plus two shims that are provably exact (B1), would remove all 18.

### B9 — Single-pass HiFT limits (confirmed). Status: **capped at 1,024 tokens / 2,048 frames, `b05bc66e8d`** (D24)
- **Per op:** at 3,200 frames, `ttnn.concat` inside HiFT needs a 1,536,032 B circular-buffer page against 1,393,440 B
  of per-core L1 (TT_FATAL). So a single pass tops out near 2,900 frames.
- **With every bucket resident:** the 09-27 cold warm-up ran HiFT through 2,048 frames, then the 2,560-frame bucket
  failed to allocate. It needed 118 MB/bank of contiguous DRAM; the largest free block was 106 MB.
- **Past the cap**, `SegmentTooLong` names the segment's length. The LLM runs one step past the cap, which tells an
  exact 1,024-token segment from a longer one. Host test: `test_segment_past_the_cap_raises`.
- Chunked HiFT would remove the cap (proposal in STATUS, D26).

### B10 — Non-streaming flow bucketing (confirmed). Status: **built, `b9f75371bc`**
- Bucketed vs exact-length: max |diff| 0.15–0.25, PCC ≥ 0.9997. The naive control (no masks): 2.4–3.9 and
  0.955–0.983.
- Gate test: `test_device_nonstreaming_bucketed_flow_matches_exact_real_checkpoint`.

### B11 — HiFT silence padding touches the tail (confirmed). Status: **measured; inaudible to WER/SIM; three listening pairs for the user**
- **09-27, on real prompt mels cut mid-speech:** the padding reaches 0.22–0.38 s back. The last 160 ms has 0.20–0.27
  log-mel L1, against the port's own 0.03–0.05. Over the whole utterance it is no larger than the port's own.
- **09-28, on Stage 1 utterances** (`~/listening`, `scripts/2026-09-28/listening_pairs.py`):
  - Where the utterance ends in near-silence, the bucketed-vs-exact difference sits at the silence's own level
    (−55 and −70 dBFS).
  - Where sound runs to the end, the difference is 27 dB below it.
- **Stage 1 WER/SIM on bucketed audio:** 0.68 % and 95.88, the reference 0.68 % and 95.21.
- `test_device_hift_bucket_padding_reach_real_checkpoint` pins the reach under 0.5 s (124 ms on its input).

### B12 — Token accuracy (confirmed). Status: **met, 96.37 %, `1333db5af5`** (D23); **holds on the larger sample: 95.94 % over 5,003 positions (B19)**
- bf16 logits: 91.10 % (max_seq_len 2,304) and 90.66 % (2,048). Every disagreement is at a small reference margin.
- fp32 logits (bf16 weights, HiFi4, fp32 accumulation): 96.37 %, for 0.3 ms per decode step. fp32 weights give the
  same 96.37 %; HiFi3 gives 96.22 %.
- **Noise floor**, the PyTorch reference in bf16 against its own fp32 run, same forced sequences: 95.70 % all-bf16,
  98.37 % with an fp32 head. So > 95 % is reachable in bf16 with almost no margin, and the head is most of it.

### B13 — A device job ignored SIGINT (observed). Status: note
- The 09-27 cold warm-up (`warmup_measure.py`) kept compiling after SIGINT and ran to completion.
- tt-metal probably installs its own SIGINT handler. Don't count on SIGINT to stop a ttnn job promptly.

### B14 — Waits on process state are unreliable in this container (confirmed). Status: **fixed in the tooling** (D25)
- PID 1 never reaps, so a finished child stays a zombie: `kill -0` and `ps -p` both report it alive. `pgrep -f`
  matched the launcher's own command line.
- Together these cost about 2.5 h of idle device time on 09-27.
- **Now** each job writes a sentinel file with its exit code as its last action. The chain waits on the file, with a
  timeout that reports and never kills (`scripts/2026-09-28/jobs.sh`).

### B15 — The bucketed start-up (confirmed). Status: **measured** (docs/VALIDATION.md, "Start-up")
- **Two processes, the first on an empty kernel cache:**

  | process | binaries compiled | warm-up | conv safety checks |
  |---|---|---|---|
  | first | 19,068 | 4,561 s | 1,628 s (includes compiling the reference convs) |
  | second, identical | 0 | 577 s | 182 s |

- **The checks rerun in every process, and they are needed.** Both processes found the same 43 disagreements:
  - 20 were real corruption of the prepared-weight fast path: `Conv1d(128->128, k=11)` at the **128** bucket
    (relative error 1.0–2.6; length 5,120 = 40 x 128 frames, **corrected 09-28**: first recorded as the 640
    bucket), the first source downsampling conv at every bucket ≥ 640 (7.7), and the second at every bucket ≥ 896
    (0.14–0.19). What they are: B17;
  - 23 were the safe-config reference's own error, arbitrated by a float64 host conv.
- **Eviction:** 416 MiB/bank with every bucket warmed. At the 3,060 conv-cache inserts free DRAM never went below
  559 MiB/bank, against the old 150 MB threshold. It is 0 in bucketed mode anyway.
- **The L1 option doesn't fit:** L1_SMALL reached 172.6 KiB after the flow set and HiFT ≤ 768 (the run stopped at
  its 160 KiB bound). The whole set would need ≥ 372 KiB, and 2,048-frame HiFT leaves ≤ 465 KiB beside its largest
  circular buffer.
- **HiFT is 77 % of the warm start-up;** its 1,792- and 2,048-frame buckets alone are 254 s.

### B16 — Stage 1 under the protocol (confirmed). Status: **RTF, token accuracy, WER and SIM met; verdicts recorded**
- **Warmed** (demo, 09-28): 0 binaries compiled, warm-up 542 s. Six distinct utterances at RTF 0.428–0.633,
  aggregate 0.481. LLM decode is 44–60 % of each request.
- **Cold first request** (`--warmup none`): 706 binaries compiled, 277.2 s for 8.52 s of audio, RTF 32.5.
- **Perf test** (pytest, enforcing `Meets()`): passed. 0 binaries compiled, so the pytest fixture allocates exactly as
  the demo does. The same tokens, warm-up 533.9 s, RTF 0.436–0.633 (aggregate 0.484), no evictions.

### B17 — The 20 corrupted conv geometries are tenstorrent/tt-metal#36487's bug (confirmed). Status: **comment redrafted 09-29, not posted** (`drafts/2026-09-29_comment_36487_prepare_conv_weights_dram_slicing.md`, which replaces the 09-28 draft; D29). The ROW_MAJOR-prepared candidate is B23.
- **In the pipeline** (`scripts/2026-09-28b/prepare_mismatch_probe.py`: real weights and inputs, the 128, 640 and
  896 buckets): 6 of 102 conv geometries are wrong with prepared weights; raw weights are right everywhere
  (≤ 0.0065). All six take DRAM inputs, which conv1d auto-slices.
- **It is not our call.** Passing the conv's own compute config to `prepare_conv_weights` makes the k=11 case far
  worse (1e28–1e29), and a matching slice config changes nothing. So there is nothing of ours to fix.
- **Standalone** (`repro_prepare_conv1d.py`, random weights), with the same slice config given to prepare and conv:
  - explicit DRAM width slicing (2 or 8) is wrong at every geometry tried, including ones auto slicing gets right;
  - an L1 input with no slicing is right;
  - `act_block_h_override=1024` (#35852's workaround) doesn't help.
- **#36487's own reproducer fails on this build:** prepared PCC 0.00035, raw 0.999912.
- **#55545** (conv1d, a band of lengths on Wormhole) is very likely the same bug, where auto slicing picks a bad
  split.
- Chunked HiFT (B18) runs only the 256- and 512-frame geometries. There the only disagreements are the safe-config
  reference's own error.

### B18 — Chunked HiFT (confirmed). Status: **built and gated, `da90d8cc84`** (D26)
- 512-frame calls with upstream's streaming cache (8-frame overlap, source carry-over, Hamming crossfade), the last
  call anchored to the end. HiFT goes from 12 geometries to 2, with no length limit; the cap is back to 1,600.
- **Seam gate** vs upstream's own streaming HiFT (reference venv, the same schedule and noise), on 600-, 1,016- and
  1,500-frame test-clean mels:
  - mechanism (F0 injected): seams PCC ≥ 0.99855, max |diff| ≤ 0.034; whole signal PCC ≥ 0.99927;
  - the no-crossfade control fails at 2 of 4 seams (the other two already agree, one near-silent);
  - own F0: log-mel L1 0.090–0.109, the same as single pass (0.09–0.12). Chunking adds nothing to the port's own
    spectral error.
- chunking.py's stitch equals upstream's `fade_in_out` stitch exactly.
- Upstream itself, chunked vs single pass: waveform PCC 0.49–0.82 (the sine phase restarts each call) but log-mel
  L1 0.011–0.039.

### B19 — Token accuracy on the larger sample (confirmed). Status: **holds, thin margin, `c7df6d00d5`** (D28)
- 27 sequences, 4 speakers (the corpus plus a 20-sequence extension): **95.94 % over 5,003 positions**. The first
  seven gave 96.37 %, case for case as before; the extension's 20 gave 95.79 %. Per sequence: 92.4–100 %.
- Noise floor on the same positions: 96.45 % in bf16, 98.58 % with an fp32 head.

### B20 — Stage 1 re-verified on chunked HiFT (confirmed). Status: **all four targets met, `7bd094cc3e`**
- **Start-up:**
  - cold (empty kernel cache): 1,831 s (30.5 min), 9,959 kernels; before chunking, 4,561 s and 19,068;
  - warm: 194.6 s (3.2 min), 0 compiled, checks 21.5 s; before, 577 s. The flow is now 157 s of it.
- **Geometries:** 8 LLM prefill + decode, 17 flow, 2 HiFT. DRAM warmed: 146.6 MiB/bank (before 416).
- **The checks caught one corrupted geometry,** in the flow's new 2,560-token bucket: the CFM's
  `Conv1d(320->256, k=3)` at length 5,120 (prepared 2.13, raw 0.0026). It is B17's bug.
- **RTF:** 0.433–0.628, aggregate 0.479. The perf test passed (worst 0.621), 0 compiled.
- **WER/SIM:** 0.68 % / 95.87 (reference 0.68 % / 95.21). Token accuracy 95.94 % (B19).

### B21 — HEAD re-verified on a second pod (confirmed, 09-29). Status: **measured**
- **Pod:** `app-5ddf2d9d-deployment-5545b7c7f8-c4vpz`, n150 L at `0000:01:00.0`, KMD 2.9.0 (O2), firmware
  19.11.0.0, 16 GT/s x16. It has no `python_env`, so everything ran in `/opt/venv` with `inflect` 7.5.0 added
  (RUNBOOK §2). Scripts and snapshots: `scripts/2026-09-29/`.
- **The reference side, rebuilt from scratch:** the venv from the committed requirements, upstream at
  `074ca6dc9e80`, the checkpoint at `eec1ae6c` (D35), and LibriSpeech test-clean (md5 checked). It reproduces the
  09-27 reference wherever that was recorded:
  - prompt tokens: 175 and 168;
  - the six cases' audio lengths, exactly;
  - per-case WER and SIM, to two decimals (corpus 0.68 %, SIM 95.21);
  - teacher-forced positions: 1,349 + 3,654;
  - the HiFT stream reference: chunking's stitch matches upstream's exactly, and chunked vs single pass is PCC
    0.489–0.822;
  - the shim self-test: 8.1e-6, against 1.8 for the control.
- **The device suite** (cold kernel cache, 16,747 binaries compiled): 204 passed and 3 skipped in 1:13:01.
  - Token accuracy: 95.94 % over 5,003 positions (203 disagreements, median margin 0.024, max 0.226). Exactly B19.
  - The seam gate: every printed digit equals VALIDATION's chunked-HiFT table.
  - TT's tokens: 213/95/347/75/317/202. Equal to B20's Stage 1 run.
  - The HiFT padding reach: 124 ms, as before.
- **The perf test**, in its own process, passed: worst RTF 0.634, aggregate 0.479. On 09-28 it gave 0.621 and
  0.490.
  - It compiled 5,214 binaries, and its warm-up took 1,309 s.
  - The suite had filled the cache, but this process allocates differently, so the kernels that carry DRAM
    addresses compiled again (B3). This start-up is neither cold nor warm.
- **The conv checks in the suite** fired 37 times and rejected the prepared weight 25 times. These were random-weight
  unit tests at short lengths, plus the deliberate corrupted-weight test.

### B22 — Two numbers in the rebuild spec conflict with the record (confirmed, 09-29). Status: **closed** (R1: B23; R6: B32)
- **#36487's own reproducer:** the spec says prepared PCC 0.225 under TILE. The pushed log
  (`scripts/2026-09-28b/repro_36487.log`) says 0.000352 for the reproducer as written.
  - **Re-measured 09-29** (`scripts/2026-09-29/r1_repro_36487.py`):
    - TILE declared, as written: 0.000768, with inf in the output;
    - ROW_MAJOR declared: 0.999912;
    - raw: 0.999912.
  - The spec's 0.225 does not reproduce. 0.000768 is the figure in VALIDATION and in the comment draft (D37).
- **"Cold first request on chunked HiFT: RTF 32.5":** exactly the pre-chunking figure already in PERF.md (277.2 s
  for 8.52 s), so it may be a copy.
  - Unverified; R6 measures it fresh.
  - R6 also re-measures PERF.md's cold and warm start-up (30.5 and 3.2 min), which weren't re-verified on 09-29
    (D37).

### B23 — #36487: the declared input layout decides it; a ROW_MAJOR-prepared candidate (confirmed, 09-29). Status: **built, `79349deacd`** (R1)
- **Standalone** (`scripts/2026-09-29/r1_prepare_layout.py`, 36 geometries with the pipeline's configs):
  - wherever the weight prepared declaring TILE is wrong (1.2–3.8), the one prepared declaring ROW_MAJOR is right;
  - wherever TILE is right, ROW_MAJOR is wrong (1.07–1.37);
  - the right one gives exactly the raw weight's error;
  - the exception is the flow CFM's `Conv1d(320->256, k=3)` at 5,120, wrong both ways (1.26 / 1.36; raw 0.0036).
  - The streaming HiFT lengths (108 and 208 frames) put every k=11 resblock conv where TILE is wrong: 6 convs, as
    the spec said.
- **Why, from the code:**
  - `conv1d` width-slices DRAM inputs by default (`conv1d.cpp:82-88`);
  - a sliced op, or a ROW_MAJOR input, gets a smaller channel alignment (`conv2d_utils.cpp:92-99`);
  - but `prepare_conv_weights` never takes the DRAM path for a 1-D conv (`prepare_conv2d_weights.cpp:1313`).
- **Built:**
  - `TtConv1d._verify_and_resolve` adds the ROW_MAJOR-prepared weight as the second of four candidates, ahead of
    the raw weight, so ties keep a prepared weight;
  - the unit test runs at two real broken geometries, where the candidate wins, tied with raw at 0.0039;
  - where the op rejects a ROW_MAJOR-prepared weight outright, the candidate is skipped and the other three
    compete. The case seen: the F0 predictor's width-sharded `Conv1d(80->512, k=3)` at 8–20 frames in the unit
    tests, which fails `TT_FATAL: act_matrix_width == weight_matrix_height` at validation, before any device work.
    Those tests pass, resolved to the raw weight as before.
- **In the pipeline** (Stage 1 demo, `scripts/2026-09-29/phase_r1verify.sh`):
  - 898 binaries recompiled, as the spec predicted;
  - five warm-up disagreements, none changing hands: the flow conv above (TILE 2.126, ROW_MAJOR 2.267, raw
    0.00265, exactly the spec's figures), and four HiFT geometries where the safe reference itself was off;
  - tokens and audio bit-identical to the baseline on HEAD;
  - RTF 0.432–0.630.
- **Rejected on 09-28** (from the rebuild spec; the run was lost, not re-measured): converting the conv *inputs*
  to ROW_MAJOR.
  - It broke 30 other convs and made HiFT 22–28 % slower.
  - HiFT PCC vs torch improved from 0.9909 to 0.9966. That may be a future accuracy lever.

### B24 — R2: the seam gate on nine voiced seams (confirmed, 09-29). Status: **built, `c2c387db2f`**
- **The set:** six test-clean mels from six speakers, each cut so that every crossfade is voiced: upstream's own F0
  above 10 Hz over the crossfade ±4 frames (`scripts/2026-09-29/r2_select_seam_mels.py`). Nine seams.
- **The metric:** the error relative to the signal over each 160 ms crossfade.
  - Mechanism: 0.041–0.078, where the spec recorded 0.049–0.078. The bound is 0.10.
  - No-crossfade control: 0.108–0.473, failing at all nine, where the spec recorded 0.116–0.640, failing at all
    nine (D39).
- **Gate changes** (D39):
  - max |diff| is printed, not gated: it follows loudness (0.11 at the loudest seam, in both arms);
  - whole-signal PCC is now ≥ 0.995: 1221-135766-0011, a high-F0 voice, measures 0.99641 over the whole signal and
    0.044 at its seam.
- **The spec's caveat does not reproduce with our measure.** The spec said that at 5 of 9 seams upstream's own two
  calls already agree within 0.03–0.10. Measured as the old call's held-back tail against the new call's head over
  the crossfade, they differ by 0.13–0.63 at all nine. So here the control's failures show real discontinuities,
  not only the crossfade's ~8 % gain (which holds by construction). The 09-28 measure is unknown.
- **On the old three mels,** the relative error at the near-silent seam is 0.154, from a tiny signal. That is why
  every seam is voiced now.

### B25 — R3: streaming stage A, offline from fixed tokens (confirmed, 09-29). Status: **built, `082fad43d6`**
- **Against upstream's own streaming of TT's Stage 1 tokens** (`scripts/streaming_reference.py`; 6 utterances, 23
  chunks, 17 seams):
  - the chunk plans are identical;
  - flow per chunk: 0.0085–0.0182 relative error (spec: 0.0085–0.018). Upstream's non-streaming control is
    0.022–0.130, further away at every middle chunk;
  - HiFT with upstream's F0 and noise: PCC 0.99921–0.99985 per chunk and 0.99900–0.99989 per seam (spec: ≥ 0.999);
  - own F0: log-mel L1 0.069–0.088.
- **#36487 at the streaming lengths:** 108, 128 and 208 frames. All 18 k=11 resblock convs (6 per length) have a
  wrong TILE-prepared weight, 1.0 up to 7.9e7 or inf. The ROW_MAJOR candidate is used at 0.0035–0.0045 (B23).
- **The final chunk's end padding** (B11's tail):
  - over the whole of 121-127105-0015's 0.68 s final chunk, PCC is 0.965; before its last 0.4 s, 0.99921;
  - in the last 0.4 s of each utterance the difference sits at −55 to −82 dBFS, at the silence's own level where
    the utterance ends in near-silence.
  - Gated on level (D38).
- **WER/SIM:**
  - our streamed audio: 1.36 % / 95.83;
  - upstream streaming on the same tokens: 0.68 % / 95.90;
  - the spec recorded 0.68 % / 95.81 vs 0.68 % / 95.89.
  - The one extra word is Whisper appending "you" after the last word of 260-123440-0010, in the end-padded tail.
    Not investigated further.
- **The encoder:** the streaming look-ahead goes in place (`context_rows`), so a chunk meets only bucket
  geometries. The encoder and flow tests pass (56 passed).

### B26 — R4: streaming stage B, interleaved with the LLM (confirmed, 09-29). Status: **built, `fa4eca1213`**
- **How it runs:**
  - `generate(on_token=...)` feeds a `StreamSession`, and a due chunk's flow and HiFT run between decode steps
    under the live decode trace;
  - `warmup_streaming()` first compiles and verifies every streaming geometry: 17 flow buckets, and HiFT at 128
    padded in front, 108, 208 and 128/256 padded at the end;
  - the trace is released before the final chunk, and nothing is alive after;
  - HiFT's state stays on the host;
  - the hop restarts per segment.
- **The noise has its own generator.** Host-side RAS sampling draws from the global RNG, so noise drawn there
  between decode steps would have changed the sampled tokens.
- **The hang check first, on KMD 2.9.0 (D36):**
  - the opt-in tracker on the CFM traces: 6 passed;
  - the interleaved test under `TT_METAL_TRACE_ALLOC_TRACKING=1`: passed, no violation, no hang (16:47).
  - The spec's "no hang" was observed on KMD 2.3.0; this is the first stage-B run on 2.9.0.
  - The tracker slows each decode-trace replay about 40x (about 0.4 s a token), so tracked timings are not
    measurements.
- **The test** (greedy, 260-123286-0014):
  - 180 tokens, 4 chunks, 3 of them ready during generation;
  - greedy streamed tokens equal the batch tokens, as the spec recorded;
  - no trace alive after;
  - the streamed audio is bit-identical to stage A's offline streaming of the same tokens and noise, as the spec
    recorded.
  - Greedy decoding runs this sentence to 180 tokens; RAS gives 75.
- **#36487 in the interleaved run:** the 108- and 208-frame k=11 convs again, with TILE-prepared weights from 6e32
  and 2.5e33 up to inf. The ROW_MAJOR candidate is chosen, as the spec said R1 would be needed for.

### B27 — R5: streaming measured; without its warm-up, streaming allocates under the live trace (confirmed, 09-29). Status: **built, `8ca78aa5a1`, `6a2ab97dde`**
- **Warm, two runs** (`demo.py --stream`; `scripts/2026-09-29/phase_r5.sh`):
  - fresh processes, both warm-ups first, the six distinct utterances, RAS seed 1986;
  - time to first audio 1.365–1.455 s and 1.336–1.479 s (target < 0.5 s);
  - streaming RTF 0.806–1.057 and 0.787–1.122 per utterance, aggregate 0.853 and 0.843 (target < 0.4). The 3.8 s
    utterance is the worst both times;
  - warm-ups: buckets 186.1 s, streaming 149.2 and 149.8 s; 0 kernels compiled; the tokens equal the Stage 1 demo's.
- **The first chunk:**
  - 0.371–0.466 s until it starts;
  - flow 0.812–0.920 s, of which the CFM takes 0.674–0.731 s (67–73 ms a step);
  - HiFT 0.121–0.127 s.

  With a free flow, first audio would be at 0.51–0.59 s.
- **The spec's R5 figures reproduce:** first audio 1.34–1.49 s; RTF 0.80–1.12, aggregate 0.85; first chunk LLM
  0.38–0.47 s, flow 0.79–0.90 s (CFM 0.66–0.70 s), HiFT 0.12 s; floor 0.50–0.59 s.
  - Today's flow and CFM run 0.01–0.03 s higher.
  - Not re-measured: "the prompt is 15–19 % of the flow" and "~65 ms per Euler step regardless of length".
- **WER/SIM** (run 1, against upstream's streaming of the same tokens): 1.36 % / 95.85 against 0.68 % / 95.90. The
  extra error is Whisper appending "you" to 260-123440-0010, as in stage A (B25).
- **Cold, under the tracker** (`--warmup none`, 121-127105-0003, `TT_METAL_TRACE_ALLOC_TRACKING=1`):
  - 416 kernels compiled. Then the first decode replay after the first chunk raised `Found 1259 device buffer(s)
    still alive before trace replay. These will be corrupted on replay.`
  - 772 of them are `ttnn.to_device` copies: weights and constants on first use.
  - 487 were allocated while new programs were created on program-cache misses, in the "program_cache: <op>"
    context of `ttnn/api/ttnn/device_operation.hpp:384` (convs 110, halos 102, moves 42, matmuls 34, others). The
    program cache keeps them.
  - The spec's "cold first request, no warm-up: 172 s to first audio, RTF 65" came from an untracked run. If that
    run took this path, it could have overwritten its own buffers. Not re-measured; the path is now refused.
  - The process exited cleanly. tt-smi afterwards: n150 L at 0000:01:00.0, DRAM OK, heartbeat 158,131, firmware
    19.11.0.0.
- **The guard (D40):**
  - `synthesize_stream` raises unless `warmup_streaming()` has run;
  - `demo.py` refuses `--stream` without `--warmup buckets` (exit 2 at argument parsing);
  - the interleaved test checks the refusal first: 2 passed, 0 kernels compiled, first audio 1.373 s, RTF 0.890
    (`phase_r5b.sh`).
- **Recorded as `Misses()`** in `tests/perf/gates.py`: `ttfp_ms` at 1470 ± 15 %, `rtf_streaming` at 1.09 ± 20 %.
  No device test enforces them yet.

### B28 — Streaming WER 1.36 %: the final HiFT call's end padding silences the last ~25 ms, and Whisper adds "you" (confirmed, 09-30). Status: **fixed, `ed1c3ad1c5`** (B29)
- **The flip:** stage A (R3) and R5's live audio both end 260-123440-0010 in "... gently smiling jaws you" (WER 5.00 % there; corpus 0.68 → 1.36 %). Upstream's streaming of the same tokens and our Stage 1 don't. `scripts/2026-09-30/`.
- **Not lengths:** all six utterances match upstream's streamed audio sample for sample in length, whole and per
  chunk. The "new frames only" slicing and the trim are exact.
- **Not the scorer:**
  - it already decodes greedily with one temperature (`temperature=0.0` as a float: no fallback), the call CosyVoice1's scorer makes;
  - 12 runs per clip (5 as run, 5 seeded with deterministic algorithms, 24 and 1 threads) gave one transcript and the same per-step log-probabilities to three decimals.
- **Where Whisper decides:** at the first text token. " how" (lowercase mode) beats " How" by 0.13 nats on ours; upstream's goes 0.54 the other way. In lowercase mode the first segment ends at 8.06 s, and a second decode over the remaining 20 ms produces " you".
- **The trigger is our last 0.1 s:** ours with upstream's last 0.1 s spliced in is clean; upstream with ours says "you".
- **What is there:** our last ~500–620 samples fall to −104 to −139 dBFS where upstream's run on at −52 to −89. The rest of the last second matches to 1–2 dB per 20 ms frame.
  - It happens in every streamed utterance. The final call is padded at its end with silence mel (`FINAL_CALL_BUCKETS`: the spec's design, D38), and HiFT's look-ahead sees the silence.
  - Stage 1 has it too wherever its single HiFT call is end-padded to a bucket: clearly in 121-127105-0003, 260-123286-0014 and 260-123440-0010. The two chunked utterances, anchored to the end, are clean. Whisper happens not to trip there (" How" by 0.27 on this clip).
- **The proof** (`you_tail_ab.py`: the final call at its exact length, nothing else changed; the padded variant reproduces R5's and R3's audio sample for sample):
  - **Mechanism** (upstream's mel, F0 and noise, so only the padding differs from upstream):
    - padded: the last 20 ms collapse; the tail difference is 0.9–23 dB below the signal; Whisper says "you" (lowercase by 0.55 nats);
    - exact: the last 20 ms within 1.3 dB of upstream's; the tail 20–29 dB below the signal; seams unchanged; no "you" (" How" by 0.575; upstream's own 0.541).
  - **Our pipeline on the clip, over 11 noise realizations:** padded says "you" 5 of 11 times, with the first token always within ±0.16 nats of a tie. Exact: 0 of 11, 0.78–1.04 nats clear.
  - **Corpus:** exact 0.68 % / 95.88; upstream 0.68 % / 95.90; padded (= R5) 1.36 % / 95.85.
- **Why 09-28 scored 0.68 %:** most likely the same end padding (the spec pads the final chunk with silence, and singles out only the first chunk for front padding), landing on the other side of the coin: 6 of 11 realizations don't tip. It can't be proven, because the lost code and its audio are gone. Either way the audio did not match upstream at the end, and the gates allowed it: D38's −50 dBFS floor passed this clip's tail at 2.5 dB below the signal.
- **Exact length isn't the product fix:** it compiled 4,046 kernels for five new final lengths (~800 each on first sight), and the final lengths are unbounded.
- **Front padding (`you_tail_front.py`) isn't either.**
  - The tails are fixed and nothing new compiles, and "you" goes (0 of 11 realizations; corpus 0.68 %).
  - But the silence under the crossfade degrades the final seam: PCC 0.9923–0.9997, five of six below the gate's 0.998. A 13-token final chunk's body falls to 0.978, and corpus SIM to 95.80.
  - That it removes "you" as well confirms the tail as the trigger.
- **Proposed** (not built; the user's call): masked end padding in HiFT, i.e. activations beyond the valid length zeroed after every layer: upstream's zero padding at the bucket geometry. It would fix the Stage 1 tails as well. It needs a gate: in the mechanism test, the final chunk's last 20 ms within 3 dB of upstream's (padded misses by 15–79 dB, exact is within 1.3), with no absolute floor.

### B29 — HiFT's padded calls masked: they compute upstream's call at the real length (confirmed, 09-30). Status: **built, `ed1c3ad1c5`**
- **Every padding site checked against upstream's code** (`tt/hifigan/valid_length.py`, D42):
  - zero-padded convs: masks after each conv and each stage's sum;
  - F0 zeroed past the end, which keeps SineGen2's interpolated phase flat as upstream's clamping does;
  - `torch.stft`'s reflect centering: the source's 8 samples after the end are reflected;
  - the iSTFT's window normalization: the last 3 samples are rescaled;
  - ReflectionPad1d((1, 0)) pads the start only, so it is untouched.
- **The host proof, with the same F0:** masked equals the real length to ≤ 1.2e-6 (8 pairs). Each rule matters:
  2.4e-5 to 4.7e-2 without it.
- **The F0 predictor alone** rounds differently at another length (≤ 7e-3 Hz). SineGen2 integrates it over the call,
  so end to end the rounding grows to ~1e-3; hence the two-part test.
- **On the device, against the exact-length call:** last 20 ms max |diff| ≤ 4.9e-4, PCC 0.9987–0.9999; whole PCC
  0.99983–1.0 (six streaming final calls, four Stage 1 calls, same and own F0). The whole-utterance max |diff| up to
  3.6e-2 is the device phase's geometry rounding.
- **D41's gate** fails on the old padding (15–79 dB under) and passes on the fix (0.2–0.8 dB): Stage 1 against
  torch, streaming against upstream.
- **Stage A's final chunks:** the last 0.4 s's difference is 21–27 dB below the signal; PCC before it
  0.99921–0.99979.
  - 121-127105-0015's whole final chunk is 0.9970 (exact length 0.9975): the port's floor at −71 dBFS. Not gated.
- **The "you" clip, 11 draws:** 0 end in "you", 0 word errors, endings within 0.2–0.6 dB of upstream's.
- **Costs:**
  - ~80 elementwise masks and a host round trip of the source per padded call;
  - 3,222 kernels recompiled once (the new allocations);
  - the final call's HiFT 0.140 against 0.120 s, measured under CPU load. The clean figure comes with the streaming
    re-measure.

### B30 — WER and similarity over five noise draws: steady, TT and reference (confirmed, 09-30). Status: **built, `5317572d0c`**
- **Protocol (D43):** seeds 1–5 for the vocoder's noise, the tokens fixed by the LLM seed.
  - The tokens are identical across every draw and equal to the undrawn runs': TT's the Stage 1 demo's, the
    reference's its 09-29 run.
  - The reference is re-seeded before each vocoder call and restored after, so its later segments sample the same
    tokens.
- **Corpus WER 0.68 % in every draw of all four groups** (TT and reference, Stage 1 and streaming). No utterance's WER
  moves between draws.
  - TT's one error (260-123440-0002, 1 word in 44) is the one upstream's streaming makes on the same tokens.
  - The reference's Stage 1 error is on its own tokens of 260-123286-0014.
- **Similarity, mean (range):**

  | | Stage 1 | streaming |
  |---|---|---|
  | TT | 95.88 (95.84–95.92) | 95.83 (95.81–95.87) |
  | reference / upstream | 95.22 (95.21–95.24) | 95.89 (95.87–95.91) |

  Per utterance it varies by at most 0.3.
- **Before the fix, the streamed clip was a coin flip:** 5 of 11 draws ended in "you" (B28). On the masked HiFT, 0
  of 11 do, and the five-draw corpus figures above hold.

### B31 — Stage 1 and streaming re-run on the masked HiFT (confirmed, 09-30). Status: **built, `ba56920c77`**
- **The device suite:** 228 passed and 4 skipped (the opt-in tracker and three reference-venv tests), in 27 min,
  with 1,103 kernels compiled.
- **The Stage 1 perf test:** worst RTF 0.675, aggregate 0.481 (Meets).
- **The Stage 1 demo:** the same tokens as 09-28/09-29; RTF 0.441–0.654, aggregate 0.483.
  - The masked padded call costs 0.07 s of HiFT at the 256 bucket and 0.10 s at 512 (about 80 masks and a host
    round trip of the source).
  - The chunked calls are unchanged.
- **The streaming demo, twice:** first audio 1.353–1.502 and 1.313–1.432 s; RTF aggregate 0.851 and 0.836, worst
  1.121 and 1.103. That is within the spread of 09-29's runs on the old padding. Both Stage 3 targets are still
  missed; the recorded bands hold every figure.

### B32 — R6: the streaming figures enforced; start-up and the cold first request re-measured (confirmed, 09-30). Status: **built, `d2a2439e05`**
- **The streaming perf test** (`tests/perf/test_pipeline_perf.py`, own process) enforces both Stage 3 figures
  through the gates: worst first audio 1,469 ms (best 1,404), worst RTF 1.110 (aggregate 0.867), both in band.
- **Start-up** (`scripts/2026-09-30/startup_measure.py`, twice against one new `TT_METAL_CACHE`):
  - `warmup_buckets()`: 1,885 s cold (9,910 kernels) and 188 s warm. That reproduces 09-28's 1,831 s (9,959) and
    195 s.
  - `warmup_streaming()`: 779 s cold (2,766 kernels) and 150 s warm.
  - So a streaming start takes 44 min cold and 5.6 min warm. Builds took 15 and 12 s.
- **The cold first request** (`demo.py --warmup none`, 121-127105-0003, 8.52 s):
  - an empty kernel cache: RTF 66.1 (563 s, 2,929 kernels; HiFT 386 s). The spec said 64.3.
  - a cache already holding its binaries (the day's earlier runs had compiled them): RTF 2.04 (17.4 s, nothing
    compiled; HiFT's first-sight conv checks 10.7 s).
  - The spec's "RTF 32.5 on a filled cache" was 09-28's cache that still lacked 706 binaries. A filled cache gives
    no fixed figure, and that closes B22.
- **Docs:** PERF.md, VALIDATION and the README carry all of it; the README got a reviewer pass (tests, layout,
  references).

### B33 — All seven N150s on this host restarted their firmware at ~16:55 on 09-29, while this card sat idle (confirmed, 09-30). Status: recorded (infra; cause unknown)
- **tt-smi's `heartbeat` for this card** (0000:01:00.0; `scripts/2026-09-29/tt_smi*.json`, `r5_tt_smi.json`,
  `scripts/2026-09-30/tt_smi_end.json`, `tt_smi_0946.json`):

  | time | heartbeat |
  |---|---|
  | 09-29 09:26:31 | 117,882 |
  | 09-29 10:00:27 | 121,913 |
  | 09-29 15:05:19 | 158,131 |
  | 09-30 09:15:09 | 116,472 |
  | 09-30 09:46:53 | 120,243 |

  It advanced at a steady 1.98/s through 09-29, idle gaps between jobs included. So 09-29's power-downs, if any,
  did not reset it.
- **The driver's counter** (`/sys/class/tenstorrent/tenstorrent!N/tt_heartbeat`) runs at 9.91/s. At 09:46:37 it read
  601,027 on this card and 601,045–601,047 on the host's six other N150s, which this session never opened.
- **Both counters date their start to ~16:55 on 09-29, ±5 min:** 601,328 / 9.91 s and 120,243 / 1.98 s back from
  09:47. Every card on the host restarted at the same moment.
- **What it rules out:** a fault of this card, and anything this session did. The last job before it ended at
  15:05 (the tt-smi check), and the next device job started at 00:13 on 09-30.
  - The host did not reboot: uptime 92 days (`/proc/uptime`).
  - The kernel log is not readable here.
  - The driver runs with `power_policy=Y`, `idle_power_down_grace_ms=5000`, `auto_reset_timeout=10` and
    `reset_limit=10`.
- **Likely a driver reload, or a reset of every card on the host. Unconfirmed.** 09-30's STATUS blamed the idle
  power-down; the steady counter through 09-29's idle gaps argues against that.
- **No job was affected.** Every 09-30 device job ran clean after it. Related to O2 only as another infra event.

### B34 — Stage A's last-0.4 s criterion was one draw from failing at 20 dB; the port's error on one burst sets it; now 15 dB (confirmed, 09-30). Status: **built, `f899ad51a2`**
- **Asked** (user, 09-30): the criterion measured 21–27 dB against 20, about 1 dB of margin. What limits the worst
  case? Set the threshold with real headroom from the measured distribution, or justify 20. It should not be one
  noisy run away from failing.
- **Measured.** `scripts/2026-09-30/tail_margin.py` runs stage A's mechanism (upstream's mel, F0 and noise per HiFT
  call) over six references: the suite's, and D43's five streaming noise draws. That is 36 final chunks per variant
  of the final call. The figures are in `tail_margin_analysis.py` → `tail_margin_analysis.md`.

  | variant | margin over the last 0.4 s |
  |---|---|
  | masked (as built) | 19.5–27.5 dB, median 26.4 |
  | exact length, unpadded and compiled per length | 19.9–29.5 dB |
  | silence padding (the code before `ed1c3ad1c5`) | −1.9 to 24.8 dB |

  At 20 dB, one draw already failed: `ref_stream_seed2`'s 121-127105-0015 measured 19.5.
- **What limits it: 121-127105-0015**, all five lowest (19.5–23.6, mean 21.6 ± 1.5).
  - Its 13-token final chunk ends near-silent: the last 0.4 s is −71 dBFS. The exception is one 20 ms burst at
    −58 dBFS, 0.38–0.36 s before the end, which holds 85–87 % of the window's signal energy and 81–94 % of its
    difference energy.
  - The window's margin is the burst's own within 1 dB, in every draw (19–24 dB). Quieter frames, some only 13–19 dB
    above their difference, carry too little energy to move it. The last 40 ms are at 22–24 dB.
  - The exact-length call scores 19.9–24.0 on the same chunks, with the burst's difference the same within about
    1 dB. So it is the port's own error on that burst, not the padding or the masking.
- **Masked against exact, per utterance:** −2.2 to +0.3 dB.
  - 121-127105-0003 and 260-123440-0010 sit 1–2 dB lower masked. Their body margins move too (26.3 against 30.1,
    26.1 against 31.2): the bucket's geometry rounds SineGen2's phase differently ("Bucketing"), and the end is not
    the cause.
- **Set to 15 dB** in `tests/e2e/test_streaming.py`: 4.5 dB below the lowest of 36, about 4.5 standard deviations
  below 0015's mean.
- **What it still catches of the old padding.**
  - On its own, 4 of the 6 utterances fail it (−1.9 to 10.9 dB).
  - 260-123286-0014 (17.8) and 260-123440-0002 (23.4–24.8) pass it. The padding's effect reaches back 120–200 ms,
    and louder speech before that dominates their last 0.4 s.
  - Both fail the 20 ms level check, by 75 and 32 dB, so the end gate as a whole fails the old padding on every
    utterance.
- **Not taken: the last 0.1 s.** Over the last 0.1 s the masked call scores 22.0–29.0 dB and the old padding −3.1 to
  0.4 dB, in all 36. A 12 dB threshold there would separate the two with 10 dB each way, where the 0.4 s window
  cannot separate them at all. Offered to the user (D41).
- **Verified:** stage A passes at 15 dB (`scripts/2026-09-30/phase_tail15.sh`).

### B35 — Stage 3, step 1: only 121-127105-0015 streams above RTF 1.0, because its tokens spill into a third chunk; the flows alone are 0.47–0.74 of every utterance (confirmed, 09-30). Status: **recorded, `3ab7d78bf0`**
- **Source:** the per-chunk records of B31's two streaming runs (`scripts/2026-09-30/rtf_breakdown.py` →
  `rtf_breakdown.md`). Each chunk's bucket is recomputed with the pipeline's own `bucket_for`.
- **Wall = text and LLM + each chunk's flow + each chunk's HiFT.** Nothing else takes more than 0.01 s.
- **0015 (3.80 s):** 1.121 and 1.103.
  - LLM 1.07 s, flows 2.81 s (0.91 + 0.89 + 1.01; CFM 2.23), HiFT 0.39 s, in run 1.
  - Its 168-token prompt is padded to 175, so the first hop is 32. After 32 + 50, its 95 tokens leave 13 for a final
    chunk: a non-streaming flow over 263 tokens (bucket 320, 1.01 s) for 0.52 s of audio.
- **260-123286-0014 (3.00 s):** 75 tokens and a 175-token prompt make exactly 25 + 50, two flows: 0.957 and 0.905.
- **The flow's cost per bucket:**
  - CFM per Euler step: 68–74 ms at 512 mel frames, 78 at 640, 86 at 768, 100 at 1,024, 135 at 1,280. Strongly
    sublinear, so a large fixed part per step.
  - The flow outside the CFM: 0.14–0.19 s at 512 frames, 1.00–1.36 s at 1,280.
- **Every utterance's flows alone come to 0.47–0.74 of its duration,** so only a cheaper flow per chunk can reach
  0.4. That means fewer Euler steps, or a cheaper step (steps 2 and 3).
- PERF's old line blamed "the ~1.4 s first chunk over the least audio" for both short utterances. The chunk count
  is what separates them.

## O: older open items

- **O1 — HiFT dtype crash.** Status: **fixed `544d588018`** (09-27). `TtHiFTDecoder.decode` converts `mel` and
  `s` to its dtype on entry. Test: bf16 generator with fp32 decoder, PCC 0.999994–0.999998 and max|diff|
  0.0004–0.0012; the unfixed code fails with the concat same-dtype TT_FATAL. The original problem: a bf16
  generator (F0 predictor and source) with an fp32 decoder failed with `TT_FATAL` in `TtStft`'s concat.
- **O2 — PCIe drops on two pods.** Status: open (infra). Both ran KMD 2.9.0. The pod with KMD 2.3.0 was stable
  throughout 09-27. Root cause unknown.
  - **09-29:** the new pod also runs KMD 2.9.0, with `power_policy=Y` and `idle_power_down_grace_ms=5000`.
  - It stayed clean through the reference rebuild, the suite and the perf test (B21).
  - The procedure if a card drops is D36.
  - It also stayed clean through R1–R5, including R5's cold run that failed in the tracker (B27).
- **O3 — `TtHiFTStreamingState` lost on 09-24.** Status: **rebuilt as `HiFTStream`** (R3, `082fad43d6`; B25).
  It follows the design in `history/BRINGUP_STATUS_25_sept.md`: mel cache 8, source cache 3840, host crossfade.
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
