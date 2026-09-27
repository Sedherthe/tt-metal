# Design: CosyVoice2 pipeline, non-streaming first (rev 2, 2026-09-27; not built)

Rev 2 revises rev 1 against the merged CosyVoice1 port (`~/reference-cosyvoice1` = PR #52540's merged head
`f8620f739c`, with no later changes on `main`): `tt/pipeline.py`, `tt/streaming.py`,
`scripts/{prepare_inputs,run_reference,eval_wer_sim}.py`, `requirements-reference.txt`, `PERF.md` §7–9 and
`tests/e2e/test_pipeline_api.py`.

It also applies the user's decisions:
- a separate frontend/reference venv;
- only `inflect` added to `python_env`;
- WER/SIM on a fixed set now, without waiting for the maintainers.

`COSYVOICE1_LESSONS.md` has not been received yet; see "Open" at the end.

## What changed from rev 1

| rev 1 | rev 2 | why |
|---|---|---|
| Eval deferred to the maintainers | **In scope.** CosyVoice1's scorer, a fixed corpus, the PyTorch reference scored by the same command | user decision; CosyVoice1 precedent |
| One `Prompt` class | `PromptContext.from_npz` (mode-aware) plus `RandomSources` | CosyVoice1 `pipeline.py`: randomness is injected, never drawn inside a forward pass |
| Modes out of scope | zero-shot first; cross-lingual and `instruct2` as prompt construction; no SFT | #54104 names no modes (below); the checkpoint has no `spk2info.pt` |
| Frontend: optional separate venv | **Separate venv, decided**, with a `requirements-reference.txt` like CosyVoice1's | user decision |
| Streaming "later" | Session, trace and carry rules fixed now, so the non-streaming build doesn't block them | CosyVoice1 `streaming.py` / `synthesize_streaming` |

## Two environments

- **`python_env` (device).** Runs the pipeline, demo and tests. It gains only `inflect`, approved; anything
  more needs asking. It imports torch, numpy, ttnn, transformers (the tokenizer) and librosa. It never imports
  onnxruntime, whisper or the upstream `cosyvoice` package.
- **`cosyvoice2_ref_env` (host-only, CPU).** Built from `requirements-reference.txt` next to an upstream
  `FunAudioLLM/CosyVoice` checkout pinned to one commit (recursive, for Matcha-TTS). It does three jobs:
  - `scripts/prepare_inputs.py`: runs the frontend over the fixed corpus and writes one flat `.npz` per case
    (text ids, prompt text ids, LLM and flow prompt speech tokens, the 24 kHz prompt mel, the CAM++
    embedding, mode and language, the checkpoint id);
  - `scripts/run_reference.py`: the upstream PyTorch `CosyVoice2` over the same corpus, writing wavs plus
    `results.json`;
  - `scripts/eval_wer_sim.py`: scores any run directory (the reference and TT alike).
- **Planned requirements** (pins chosen and measured at build time, then frozen with a header explaining each
  pin, as CosyVoice1 does):
  - `--extra-index-url https://download.pytorch.org/whl/cpu`, then `torch==X+cpu` and `torchaudio==X+cpu`
    (same X);
  - `openai-whisper`: the log-mel for the speech tokenizer, and Whisper large-v3 for WER;
  - `onnxruntime`, **pinned**: CosyVoice1 measured a different token sequence for the same audio after an
    upgrade;
  - `transformers` (Qwen2 for the reference, `WavLMForXVector` for SIM), `inflect`, `librosa`, `soundfile`,
    `wetext` (optional);
  - whatever the upstream CosyVoice2 import closure needs (HyperPyYAML, conformer, diffusers, lightning,
    hydra-core, omegaconf, ...), established by tracing imports and not copied blindly;
  - no GPU packages.
- **The torch version sets RAS's RNG stream.** CosyVoice1 found `torch.multinomial`'s single-sample path
  changed between 2.6 and 2.8. The reference venv's torch and `python_env`'s 2.11 will therefore never agree on
  seeded sampling. Cross-environment token comparisons are **teacher-forced** (R3/R8), and the reference's
  goldens are tied to its pinned torch.

## Package layout (mirroring CosyVoice1)

```
tt/pipeline.py        CosyVoice2TTNN (stage methods + synthesize; synthesize_streaming later)
tt/prompt.py          PromptContext.from_npz, RandomSources, MODES, describe_mode
tt/text.py            normalize() / split(): spell_out_number (inflect) + a verbatim split_paragraph port
scripts/              prepare_inputs.py, run_reference.py, eval_wer_sim.py   (reference venv only)
demo/demo.py          argparse CLI (below)
tests/e2e/            test_pipeline_api.py (public API), test_text.py (host), test_scoring.py (normalizer)
requirements-reference.txt, docs/VALIDATION.md, PERF.md, README.md
```

The dated `scripts/perf_*` and `vocoder_debug_*` move to the notes branch (R15).

## API

```python
MODES = ("zero_shot", "cross_lingual", "instruct2")   # no "sft": CosyVoice2-0.5B ships no spk2info.pt

@dataclass
class RandomSources:          # CosyVoice2's draws; CFM noise is the model's fixed rand_noise buffer
    sine_noise: torch.Tensor | None = None         # [1, T_audio, 9], SineGen2's per-call noise
    llm_seed: int | None = None                    # RAS; seeded host sampling (see the torch-version note)
    def sine_noise_for(self, audio_len): ...       # captured array if given, else a fresh draw
    # SineGen2's rand_ini is not modelled: never read through the downsample (test_sine_gen2.py:111).

@dataclass
class PromptContext:          # one .npz from prepare_inputs.py; mode decides which fields are set
    mode: str; lang: str
    prompt_text_ids: torch.Tensor | None           # zero_shot: transcript; instruct2: instruct text; cross_lingual: None
    llm_prompt_speech_tokens: torch.Tensor | None  # zero_shot only
    flow_prompt_speech_tokens: torch.Tensor        # all three modes (flow always conditions on the prompt audio)
    prompt_feat: torch.Tensor                      # 24 kHz mel, aligned so feat == 2 x tokens (upstream's rule)
    embedding: torch.Tensor                        # CAM++ x-vector
    @classmethod
    def from_npz(cls, path) -> "PromptContext": ...

class CosyVoice2TTNN:
    def __init__(self, device, config: CosyVoice2Config): ...          # loads llm.pt / flow.pt / hift.pt
    def text_to_tokens(self, ctx, text_ids, *, rng=None, on_token=None) -> list[int]
    def tokens_to_mel(self, tokens, ctx) -> ttnn.Tensor                # non-streaming flow (finalize)
    def mel_to_wav(self, mel, mel_frames, *, rng=None) -> ttnn.Tensor   # HiFT: F0/source fp32 by default (O6), decoder fp32
    def synthesize(self, ctx, text, *, rng=None) -> Synthesis           # normalize + split + per-segment stages
    def warmup(self, ctx, lengths) -> None                              # compile + resolve geometries before timing
    def release(self) -> None
```

- **`CosyVoice2Config`** is as in rev 1: every switch explicit, with a `reported()` preset. One change: the
  F0/source dtype defaults to **fp32**. O6 measured bf16 F0 at about 3x the Hz error (0.70 vs 0.23 Hz mean) and
  more voiced/unvoiced flips; bf16 remains selectable, and `544d588018` makes it work. Module constructors
  get keyword arguments that default to today's environment-reading functions.
- **`Synthesis`** carries the audio, per-segment tokens, mel frames and timings (LLM prefill and decode,
  encoder, CFM, HiFT; device-synchronized), plus the config.
- **`text_to_tokens`** sizes nothing itself. `max_seq_len` is fixed at construction from the config's budget,
  and `generate()` refuses overruns (R7).

**Modes are prompt construction, not networks** (CosyVoice1's table, adapted to upstream CosyVoice2's
`frontend_*`):

| mode | LLM prefix: prompt text / prompt speech | flow prompt (tokens + mel) | embedding |
|---|---|---|---|
| `zero_shot` | transcript / yes | yes | prompt audio |
| `cross_lingual` | — (target text carries a `<\|en\|>`-style tag) / — | yes | prompt audio |
| `instruct2` | `instruct_text` / — | yes | prompt audio |

## Text normalization and splitting (English path = upstream's)

- The order is:
  1. optional wetext `EnNormalizer` (skipped, as upstream skips it when wetext is absent);
  2. `spell_out_number` (inflect);
  3. `split_paragraph(tokenize, "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)`,
     ported verbatim;
  4. drop punctuation-only segments.
- Upstream also returns the text unsplit when it contains `<|...|>` markers.
- Chinese normalization (`contains_chinese` branch, wetext `ZhNormalizer`) is out of scope until a non-English
  mode is taken on.
- `test_text.py` checks the ported functions against upstream's own examples (host only).

## Evaluation (in scope now)

- **`scripts/eval_wer_sim.py`**, a port of CosyVoice1's scorer:
  - Whisper **large-v3** ASR, and **`microsoft/wavlm-base-plus-sv`** (`WavLMForXVector`) cosine × 100 against
    the prompt wav for SIM;
  - CAM++ cosine reported **only as a diagnostic**: it is self-referential, because the model conditions on
    CAM++;
  - NFKC, lowercase, punctuation stripped, word-level edit distance;
  - per-utterance and **corpus-level** WER, and `--baseline` to diff a TT run against the reference run.
- **Corpus:** one fixed definition imported by both `prepare_inputs.py` and `run_reference.py`, as CosyVoice1
  does. **Proposed:** two prompt speakers (upstream's `asset/zero_shot_prompt.wav` with its transcript, and
  `asset/cross_lingual_prompt.wav`), a fixed list of English sentences, and one seed. This is pending
  confirmation against the lessons doc's "fixed two-speaker set".
- **Token accuracy:** teacher-forced top-1 agreement over full zero-shot sequences (with the speech prompt),
  reference logits vs TT (R3/R8). CosyVoice1 reports 99.04% teacher-forced.
- **Own-F0 waveform quality** uses energy-envelope PCC and RMS ratio (CosyVoice1: 0.9975 and within 6%) plus a
  spectral distance (D16). Waveform PCC is used only with torch F0 injected.

## Room for streaming (decided now, built in step 3)

1. **Warm before trace capture.** A streaming session starts with a throwaway pass over every mid-stream
   geometry before `generate()` captures its decode trace:
   - encoder and CFM buckets up to the configured maximum, and HiFT chunk shapes (8-frame cache + 2×hop);
   - this compiles kernels and runs the conv resolver's per-geometry verification with no trace live.
   - The final chunk runs after generation ends, when the LLM trace has already been released, so it needs no
     warming.
   - This is CosyVoice1's `synthesize_streaming` warm-up chunk. Their un-warmed geometries produced wrong audio.
2. **Host-parked carried state.** The HiFT caches (8 mel frames, 3,840 source samples, 3,840 speech samples)
   stay on the host between chunks. Nothing persistent is allocated on device while the decode trace is live,
   and the Hamming crossfade is host work anyway (D8). The flow carries no state: each chunk recomputes the
   growing prefix, and only the token list grows.
   - CosyVoice1 instead used persistent device carry buffers allocated in the warm-up; its host variant wedged a
     Blackhole perf test. A switch to device buffers is kept for that case.
3. **Trace schedule.**
   - Only the LLM decode trace is live during a stream; the streaming CFM runs eager (D5).
   - The flow and HiFT run from the decode loop's `on_token` callback, as in CosyVoice1.
   - Transient tensors are freed before the next decode replay. The conv resolver's first-sight verification
     must not run during the stream (it is done in the warm-up; unexpected geometries raise instead of falling
     back).
   - This is the part CosyVoice1 still had open defects in, so it gets the most tests.
4. **Public-API tests** (`tests/e2e/test_pipeline_api.py`, driven from `PromptContext.from_npz`):
   - streaming generates the same tokens as batch (greedy);
   - consecutive utterances with one flow length replay correctly;
   - a stream leaves no trace alive;
   - each mode's prefix assembly matches upstream's `frontend_*` field set.

   These test the wiring, which per-stage tests can't see.

## Demo

```bash
python models/demos/audio/cosyvoice2/demo/demo.py --inputs <dir of .npz from prepare_inputs.py> --out out/ \
    [--modes zero_shot,cross_lingual] [--seed N] [--warmup] [--config reported|eager]
```

- Without `--inputs`, it uses one committed small prompt `.npz`, so it runs without the reference venv.
- Output: wavs plus `results.json` (the schema `eval_wer_sim.py` reads) and a Markdown table of per-segment
  timings and per-utterance RTF on distinct utterances (D14), warm or cold, and the config.

## Build order (step 2)

1. `inflect` into `python_env`.
2. The reference venv, the requirements file and the pinned upstream checkout.
3. `prepare_inputs.py` and `run_reference.py`, with the corpus.
4. `tt/text.py` and `tt/prompt.py`, with host tests.
5. `tt/pipeline.py` (non-streaming, zero-shot), the demo, and the public-API test for `synthesize`.
6. `eval_wer_sim.py`: score the reference run, then the TT run, and diff them. Teacher-forced token accuracy.
7. Cross-lingual and `instruct2` (prompt construction plus tests).

## Open

- **`COSYVOICE1_LESSONS.md` not received.** The corpus choice and any lesson not visible in CosyVoice1's code
  are pending it.
- The pins for the reference venv are measured at build time.
