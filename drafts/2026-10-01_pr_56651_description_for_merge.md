<!-- DRAFT for the user. The description for tenstorrent/tt-metal#56651 while it sits in the merge queue: main
squash-merges with the PR's title and description as the commit message, and today's description is the 09-15 text
("This is a draft ... not a final submission"). Every figure is the merged head's, `33aa3601eb` (README.md, PERF.md,
docs/VALIDATION.md at that commit), N150, 2026-09-30. Nothing from the backup's three later commits is claimed. -->

### Summary

TTNN bring-up of [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) in `models/experimental/cosyvoice2/`. Relates to #54104.

The whole model runs on an N150 (Wormhole): text → speech tokens (Qwen2 LLM) → mel (flow matching) → 24 kHz waveform (HiFT vocoder).
- **Stage 1:** the four targets are met.
- **Streaming:** runs on upstream's chunk schedule while the LLM generates, and matches upstream's own streaming run on the same tokens.
- **Stage 3:** both targets are missed, and both are measured: where the time goes, one CFM Euler step profiled, and the Euler step count swept. Work on them continues in a follow-up PR.

### Targets (N150, 2026-09-30)

| target (#54104) | stage | measured | status |
|---|---|---|---|
| RTF < 1.0, non-streaming | 1 | worst 0.654, aggregate 0.483, six distinct utterances | met; `tests/perf/test_pipeline_perf.py` |
| token accuracy > 95 % | 1 | 95.94 %, teacher-forced over 5,003 positions (27 sequences, 4 speakers) | met; `tests/e2e/test_token_accuracy.py` |
| WER < 5 % | 1 | 0.68 % in each of five vocoder noise draws; the PyTorch reference also 0.68 % | met |
| speaker similarity > 0.60 | 1 | 0.959 (cosine); the PyTorch reference 0.952 | met |
| time to first packet < 500 ms | 3 | 1.31–1.50 s | missed; held in a recorded band by the streaming perf test |
| streaming RTF < 0.4 | 3 | worst 1.10–1.12, aggregate 0.84–0.85 | missed; held in a recorded band |

- WER: Whisper large-v3. Similarity: `microsoft/wavlm-base-plus-sv` x-vector cosine. One script scores this port and the PyTorch reference.
- The corpus is small: six LibriSpeech test-clean utterances from two speakers.

### What's in the package

- `tt/pipeline.py`: `CosyVoice2TTNN.synthesize` and `synthesize_stream`, the reported configuration, bucketing, and the start-up warm-ups.
- `tt/llm/`, `tt/flow/`, `tt/hifigan/`: the Qwen2 LLM (on tt_transformers), the Conformer encoder and the CFM, and HiFT. HiFT includes the iSTFT as a matmul plus a transposed conv, since TTNN has no FFT.
- `tt/streaming.py`: upstream's chunk schedule, and HiFT with upstream's streaming cache and crossfade.
- `tt/text.py`: upstream's English text frontend, with parity tests.
- `demo/demo.py`: the Stage 1 demo; `--stream` for streaming.
- `tests/`: per-module PCC tests against torch or upstream, e2e tests, and the perf gates.
- `README.md`, `PERF.md`, `docs/VALIDATION.md`: how every figure was produced.
- The device side and the PyTorch reference run in separate venvs and exchange files only (`docs/security.md`).

### Known issues

- **#36487 (`prepare_conv_weights` with DRAM-sliced conv1d):** some HiFT and flow convs hit it at particular lengths. Every new conv geometry is checked once against a raw-weight, safe-config reference, with a float64 host conv as arbiter, so a wrong fast path never reaches the output.
- **Conv config tensors in DRAM:** their addresses become compile-time arguments, so cached kernels are reused only by a process that allocates identically. Start-up runs a fixed warm-up: 3.1 min with the kernels on disk, 31.4 min on an empty cache, plus 2.5 and 13.0 min for the streaming set.
- **Streaming needs its own warm-up.** `synthesize_stream` refuses to run without `warmup_streaming()`.
- **Blackhole is untested.**

### Testing

- **On an N150, 2026-09-30.** The commits after these change docs only.
  - The suite, at `5317572d0c`: 228 passed, 4 skipped (the opt-in tracker test and three reference-venv tests),
    with the two perf tests deselected.
  - The Stage 1 perf test passed (worst RTF 0.675).
  - The streaming perf test passed inside its bands (`d2a2439e05`).
  - Stage A passed at its 15 dB threshold (`f899ad51a2`).
- **Tier-3 CI, `wh_n150` (run 36746885220, by @mtairum):** the real-checkpoint PCC tests passed, 18 tests.

### Next (follow-up PR)

- The CFM's attention heads merged in one op (`nlp_concat_heads`): bit-identical output, and device time per Euler step 47.8 → 30.0 ms.
- The Euler step count as a config option; the default stays upstream's 10.
- A traced CFM step during streaming, which is proposed and prototyped.
