# Design proposal: non-streaming `tt/pipeline.py` (2026-09-27, not built)

This is order-of-work step 2 in STATUS. The streaming loop (step 3) will be built on the same objects, so the API
leaves room for it.

## Goals

- **One entry point** from text plus a speech prompt to 24 kHz audio, matching upstream's
  `CosyVoice2.inference_zero_shot(stream=False)` segment for segment.
- **Explicit configuration.** No environment variables change behaviour behind the caller's back, and every
  result carries the configuration that produced it (D14).
- **A timing breakdown per segment** (LLM, encoder, CFM, HiFT; device-synchronized), so RTF comes from distinct
  utterances (D14).
- **No new dependencies in the device venv** unless the user approves them. The heavy frontend stays behind a
  boundary, following CosyVoice1's accepted precedent.

## API

```python
@dataclass(frozen=True)
class CosyVoice2Config:
    # model
    repo_id: str = "FunAudioLLM/CosyVoice2-0.5B"  # checkpoints via HF_HOME
    euler_steps: int = 10  # D4
    flow_dtype: ttnn.DataType = ttnn.bfloat16
    hift_decoder_dtype: ttnn.DataType = ttnn.float32  # D6
    hift_source_dtype: ttnn.DataType = ttnn.bfloat16  # F0 predictor and NSF source (the item 4 boundary)
    # performance switches; defaults = the configuration we report
    llm_decode_trace: bool = True
    cfm_trace: bool = True  # non-streaming only; keyed on exact length
    flow_fused_sdpa: bool = True
    flow_fused_qkv: bool = True
    flow_matmul_accurate: bool = False
    # LLM sampling (upstream defaults)
    sampler: str = "ras"  # "ras" | "greedy"
    top_p: float = 0.8
    top_k: int = 25
    win_size: int = 10
    tau_r: float = 0.1
    min_token_text_ratio: float = 2.0
    max_token_text_ratio: float = 20.0
    # context budget: sizes ModelArgs.max_seq_len once, at construction (R7)
    max_prompt_speech_tokens: int = 750  # 30 s of prompt at 25 Hz (upstream's own prompt-length limit)
    max_segment_text_tokens: int = 100  # split_paragraph's 80-token segments plus slack
    # text frontend (upstream split_paragraph parameters)
    text_frontend: bool = True
    token_max_n: int = 80
    token_min_n: int = 60
    merge_len: int = 20
    seed: int | None = None
```

- **Construction.** `CosyVoice2Pipeline(device, config)` loads `llm.pt`, `flow.pt`, `hift.pt` and the BlankEN
  tokenizer. It builds `TtQwen2LM` with `max_seq_len = required_max_seq_len(prefix_budget, max_tokens_budget)`
  from the config, and it builds the flow and HiFT modules.
- **`prepare_prompt(prompt: PromptInputs) -> Prompt`.** Uploads the cached prompt features: prompt text ids,
  speech tokens, 24 kHz mel `prompt_feat`, and the CAM++ embedding. The result is reused across many texts.
  `Prompt.from_npz(path)` / `Prompt.save(path)` define the frontend boundary (below).
- **`synthesize(text, prompt, *, seed=None) -> Synthesis`.**
  - `Synthesis.audio`: np.float32, 24 kHz, all segments concatenated.
  - `Synthesis.segments`: `[SegmentResult(text, tokens, mel_frames, audio, timings)]`.
  - `Synthesis.config`: the config that produced it.
  - `timings`: prefill, decode, encoder, CFM and HiFT seconds, with `llm_tokens` and `audio_s`.
- **`warmup(lengths)`.** Optional: compiles kernels and resolves conv geometries for the given lengths, so
  measured runs aren't first-sight runs (D7). Warm and cold are reported separately.
- **`release()`.** Frees traces (LLM, CFM) and the geometry caches.

## Mapping to upstream (`cosyvoice/cli/cosyvoice.py`, `frontend.py`, `model.py`, fetched 2026-09-27)

| Upstream | Pipeline |
|---|---|
| `frontend.text_normalize(prompt_text, split=False)` | `normalize(prompt_text)` |
| `frontend.text_normalize(tts_text, split=True)`. English path: optional wetext `EnNormalizer`, then `spell_out_number` (inflect), then `split_paragraph(..., "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)`, then drop punctuation-only segments | `segments = split(normalize(text))`: a verbatim port of `split_paragraph` and `is_only_punctuation` from `cosyvoice/utils/frontend_utils.py` (pure Python), plus `spell_out_number` |
| `frontend_zero_shot`: text tokens, prompt text tokens, speech tokens (`speech_tokenizer_v2.onnx` on whisper log-mel), `speech_feat` (24 kHz mel), CAM++ embedding; `token_len = min(feat//2, tokens)` alignment | `Prompt` (precomputed; the alignment happens in `prepare_prompt`) |
| `model.tts(stream=False)` per segment: LLM `inference` with `min_len = 2·text_len`, `max_len = 20·text_len`, RAS | `TtQwen2LM.generate(..., min_tokens, max_tokens, sampler)` |
| `token2wav(finalize=True)`: `flow.inference` (streaming=False), then `hift.inference` | `TtCausalMaskedDiffWithXvec.inference`, then `TtHiFTGenerator.inference` |
| per-segment `yield` | `Synthesis.segments`, concatenated into `audio` |

**Deviation, documented:** no `speed` argument (upstream's time-stretch of the mel). Cross-lingual and instruct
modes are out of scope; the bounty doesn't ask for them.

## Frontend boundary and dependencies

| Need | Package | In the device venv? |
|---|---|---|
| Qwen2 tokenizer | `transformers` | present |
| mel basis, resampling | `librosa`, `soundfile` | present |
| number spelling (`spell_out_number`) | `inflect` | **missing; small pure-Python install, needs approval** |
| wetext English normalizer | `wetext` | missing. Optional upstream; propose skipping, as upstream does when it isn't installed |
| speech tokens and CAM++ embedding | `onnxruntime` plus the two ONNX files | **missing; install needs approval**, or keep it outside the venv |
| whisper log-mel for the tokenizer | `openai-whisper` | missing. Pulls in torch, so it would need the CPU index. Propose a torch/librosa reimplementation instead (checked once against whisper) |
| kaldi fbank for CAM++ | `torchaudio` | missing. Must match torch 2.11.0+cpu from the CPU index. Or reimplement |

**Proposal:** follow CosyVoice1.
- Audio-prompt features are computed by `scripts/prepare_prompt.py`, which runs in a small separate venv with
  `onnxruntime`, `openai-whisper` and `torchaudio` (CPU). It writes a `Prompt` `.npz`, and the device venv never
  imports those packages.
- Text normalization and splitting run at synthesis time inside the pipeline. That needs only `inflect`, which
  requires your approval.
- The alternative is to install `onnxruntime` and CPU `torchaudio` into `python_env` and reimplement the whisper
  log-mel. That is one venv, but three installs.

## Configuration: explicit, not environment variables

- Today, `COSYVOICE2_FLOW_SDPA`, `COSYVOICE2_FLOW_FUSED_QKV`, `COSYVOICE2_FLOW_CFM_TRACE`,
  `COSYVOICE2_FLOW_ENCODER_TRACE`, `COSYVOICE2_FLOW_MATMUL_CC` and `COSYVOICE2_CONV_CONFIG_IN_DRAM` are read inside
  constructors, and `use_decode_trace` defaults to off.
- **Proposal:** each module constructor gains keyword arguments (`fused_sdpa=`, `fused_qkv=`, `use_trace=`,
  `matmul_accurate=`, ...) that default to the current environment-reading functions. Scripts keep working; the
  pipeline passes every value from `CosyVoice2Config`, so the environment can't change a pipeline run.
- `CosyVoice2Config.reported()` is the configuration behind published numbers. Each `Synthesis` carries its
  config, and the demo prints it.

## Demo (`demo/demo.py`, argparse, like CosyVoice1)

```bash
python models/demos/audio/cosyvoice2/demo/demo.py \
  --prompt prompts/librispeech_dummy_0.npz \
  --text "Please close the door when you leave." --text-file more.txt \
  --out out/ [--seed 0] [--warmup] [--config reported|eager] [--steps 10]
```

- **Writes:** one wav per input (segments concatenated), plus `summary.json` and a Markdown table: per-segment
  text, tokens, audio seconds, per-stage timings, RTF per utterance (distinct utterances, D14), warm or cold, and
  the full config.
- **Default prompt:** a pre-built `Prompt` for LibriSpeech-dummy item 0 (the pair used so far), so the demo runs
  without the frontend venv.
- **Out of scope:** `--eval` (WER/SIM) waits for the maintainers' answer on the evaluation set (D13).
- **Tests:**
  - `test_pipeline.py`: normalization and splitting against upstream examples (host), and the budget
    arithmetic (host).
  - One real-weight device smoke test: a short text produces audio of the expected length, with per-stage
    timings recorded.
