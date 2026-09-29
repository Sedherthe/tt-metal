# Streaming design (rebuilt 2026-09-29)

The 09-28 design note was lost with that pod. This rebuilds it from the rebuild spec (`REBUILD_2026-09-29.md`,
R3–R5), with every statement about upstream re-checked in its source (`FunAudioLLM/CosyVoice` at `074ca6dc9e80`) and
every statement about our code checked in the tree. Numbers from the 09-28 runs are claims until R3–R5 re-measure
them; they are listed last.

## What upstream does (`cosyvoice/cli/model.py`, `CosyVoice2Model`)

- **The loop** (`tts`, :343-374).
  - The LLM runs in a thread and appends tokens.
  - The main loop emits a chunk whenever at least `hop + pre_lookahead_len` (3) tokens past `token_offset` exist:
    `token2wav(tokens[: offset + hop + 3], offset, stream=True, finalize=False)`.
  - **The first hop** is `25 + prompt_token_pad`, where `pad = ceil(P / 25) * 25 - P` for a P-token prompt. So the
    first chunk ends on a 25-token boundary of prompt + speech: the flow's chunk-causal mask aligns.
  - **After each chunk** the hop doubles, up to 100 (`stream_scale_factor` 2, `token_max_hop_len` 100): 25 (+ pad),
    then 50, then 100, 100, ...
  - `token_hop_len` is an instance attribute that `tts` never resets (:360), so a second request starts at hop 100.
    We reset it per utterance (D3).
- **The final chunk** (:365-374). After the LLM ends, `token2wav(all tokens, offset, finalize=True)` passes no
  `stream`, so the flow runs `streaming=False` (:292, :301): full attention over every token (D31).
- **The flow per chunk** (`CausalMaskedDiffWithXvec.inference`):
  - it gets the prompt plus every token so far, and recomputes the whole prefix;
  - with `finalize=False`, the last 3 tokens are `context` for the encoder's pre-look-ahead layer only;
  - encoder and CFM run with chunk-causal masks;
  - it returns the mel of every generated token, and `token2wav` keeps the frames from `2 x offset` (:303).
  - The CFM noise is a fixed buffer indexed from position 0, so a prefix frame sees the same noise in every chunk.
- **HiFT per chunk** (`token2wav`, :304-326):
  - the new frames are prefixed with the previous chunk's last 8 mel frames (`mel_cache_len`);
  - `hift.inference(mel, cache_source)` overwrites the first 3,840 source samples with the previous chunk's last
    3,840, so the sine phase runs on;
  - the output's first 3,840 samples are crossfaded with the previous chunk's held-back last 3,840, using
    `fade_in_out` and `np.hamming(7680)`;
  - a non-final chunk holds its last 8 frames, 3,840 source samples and 3,840 output samples back.
  - So the HiFT lengths are:

    | chunk | frames |
    |---|---|
    | first | `2 x (25 + pad)` = 50–98, no cache |
    | middle | `8 + 2 x hop` = 108 or 208 |
    | final | `8 + 2 x (tokens left)` |

  - The crossfade's two Hamming halves sum to 1.0798–1.0800. Where the two calls agree, the overlap therefore gets
    about 8 % gain (checked 09-29).

## Ours

### Stage A: offline streaming from fixed tokens (R3; no LLM, no trace)

**A `StreamSession`** drives upstream's schedule over a fixed token list. It holds:
- the tokens and the offset;
- the hop, starting at 25 plus the prompt pad and reset per utterance;
- the HiFT state on the host (about 30 KB): the mel cache, source cache and held-back speech;
- the noise source.

**The flow, per middle chunk:**
- Tokens: the prompt, then the generated tokens up to `offset + hop`, then the 3 look-ahead tokens *in place*,
  right after the valid rows, all padded to a bucket from the existing non-streaming set.
- The encoder must not create a conv geometry per chunk. Today its streaming path runs `pre_lookahead_layer` over
  exactly `valid_length` rows plus a separate context: a new geometry for every chunk length.
  - With the context in place, `pre_lookahead_layer` runs at the bucket length, like the non-streaming bucketed
    path, and the rows past `valid + 3` are zeroed.
  - The valid rows then see exactly what upstream computes: the look-ahead conv reads the context rows, and conv2
    is causal.
  - The chunk-causal mask's key-padding term hides the context and padding rows from attention.
- The CFM runs `streaming=True` over the bucket with the valid mask.
- The chunk emits mel frames `[2 x offset, 2 x (offset + hop))` of the generated part.

**The flow, final chunk:** the existing bucketed non-streaming flow over all tokens, already warmed. It emits the
frames from `2 x offset` (D31).

**HiFT:**
- Middle chunks run at exactly 108 and 208 frames.
- The first chunk (50–98 frames) is padded to 128 with silence **in front**.
  - Its last 8 frames are held back and crossfaded, so they must see upstream's right context: the conv's own
    zero padding, not silence frames.
  - Front padding is unvoiced. The F0 predictor gives ~0 there, so the sine phase at the first real frame is what
    upstream has.
- The final chunk is padded at the end to 128 or 256 (the tail effect of B11).
- Source carry-over and the crossfade reuse `tt/hifigan/chunking.py`'s pieces.
- Check the emitted audio of the padded chunks against upstream, not only the seams. Silence padding reached back
  0.22–0.38 s in the bucketing test (B11), more than the 8 held-back frames.

**New geometries:**
- HiFT at 108, 128, 208 and 256 frames. At 108 and 208, every k=11 resblock conv hits #36487; R1's ROW_MAJOR
  candidate keeps them prepared (FINDINGS B23).
- The flow's streaming masks at the non-streaming buckets: the same conv shapes, and the masked-SDPA programs the
  streaming CFM already uses (X10).
- All of them must be warmable at start-up, so that stage B can warm them before the decode trace exists.

**The gate** runs against upstream's own streaming run on the same tokens, in the reference venv, with the hop
reset per utterance on both sides:
- **Per-chunk mel.** Upstream's non-streaming mel is a control that must differ.
- **HiFT on upstream's mels,** with upstream's F0 injected and the same noise, checked at each seam.
- **Spectral metrics** with our own F0.
- **WER and SIM** on the six streamed utterances.

### Stage B: interleaved with the LLM (R4)

- **Driving:** `generate(on_token=...)` feeds a `StreamSession` as tokens arrive. The hop resets to 25 per
  utterance.
- **The trace schedule:**
  - every streaming geometry is compiled and verified before the LLM decode trace is captured, so no conv safety
    check runs while the trace is alive;
  - the trace is released before the final chunk (D22);
  - nothing is alive after a call.
  - A trace kept across the flow and the vocoder hung the card on 09-21 (RUNBOOK §4).
- **KMD 2.9.0 (D36):** first a hang check (the opt-in allocation tracker, then one short interleaved run), then the
  full gate. If the card drops, stop and report.
- **Tests:**
  - at least one chunk is emitted during generation;
  - greedy streamed tokens equal the batch tokens;
  - no trace is left alive.

### Stage C: measurement (R5)

- **Warm:** six distinct utterances, two runs.
- **Cold:** separately.
- **Per stage, for the first chunk:** the LLM decode of its tokens, the flow (with the CFM separately), and HiFT.
- **Targets** (#54104 Stage 3): time to first packet < 500 ms, streaming RTF < 0.4.

## The 09-28 numbers (claims until re-measured)

- **R3:**
  - mel per chunk: 0.0085–0.018 relative error;
  - vocoder seams: PCC ≥ 0.999;
  - WER/SIM: 0.68 % / 95.81, against upstream's 0.68 % / 95.89.
- **R4:** no hang on N150 with KMD 2.3.0 (this pod runs 2.9.0); streamed audio bit-identical to offline streaming;
  greedy tokens equal to batch.
- **R5:**
  - time to first audio: 1.34–1.49 s;
  - streaming RTF: 0.80–1.12 (aggregate 0.85);
  - cold first request: 172 s to first audio, RTF 65.
  - The first chunk:
    - LLM decode of 28–35 tokens: 0.38–0.47 s;
    - flow: 0.79–0.90 s, of which the CFM is 0.66–0.70 s;
    - HiFT: 0.12 s.
  - The CFM costs about 65 ms per Euler step at 128 and at 512 frames.
  - Decode plus HiFT alone put first audio at 0.50–0.59 s: the first chunk needs 25 tokens, the model's trained
    chunk size.
