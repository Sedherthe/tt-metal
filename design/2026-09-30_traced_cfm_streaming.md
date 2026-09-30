# Design: the CFM's Euler step traced during streaming (proposal, 2026-09-30; not built)

The user asked for this proposal, not the build (09-30, lever (b)). The evidence comes from a throwaway prototype in
the notes (`scripts/2026-09-30/traced_cfm_prototype.py`); the PR is unchanged.

## Why

- **The eager step is host-bound.** At the first chunk's size (512 mel frames, batch 2) the host takes ~61 ms to
  enqueue a step's 1,102 ops. Since lever (a), the device needs 30 ms of kernel time for them (B36, B40).
- **So device-side wins show only once the step is traced:**

  | mel frames | eager step | traced step | eager solve (10 steps) | traced solve |
  |---|---|---|---|---|
  | 512 (the first chunk's bucket) | 62.4 ms | 31.3 ms | 0.64 s | 0.32 s |
  | 768 | 64.9 ms | 52.8 ms | 0.64–0.65 s | 0.54–0.55 s |
  | 1,024 | 67.1 ms | 62.1 ms | 0.67–0.72 s | 0.64 s |

  (After lever (a); `scripts/2026-09-30/cfm_step_profile.py`, `merge_time.json`.)

## The constraint

- **A streamed chunk runs between two LLM decode steps, while the LLM's decode trace is alive** (D40, D44). So its CFM
  traces and the decode trace would all be alive together.
- **What the allocation tracker flags** (`tt_metal/impl/allocator/trace_allocation_tracker.cpp`):
  - a trace is registered at `end_trace_capture` and unregistered at `release_trace`;
  - from then on, every non-trace buffer allocated is recorded against it;
  - at each `execute_trace`, any recorded buffer still alive fails the replay.

  The check is conservative: it does not look at addresses. A buffer allocated after a capture may sit in memory
  that trace's intermediates used, and the replay would overwrite it.
- **What that forces:**
  1. every persistent buffer of every trace exists before the first capture;
  2. no capture allocates anything that outlives it, so outputs are written into pre-allocated buffers;
  3. every program the traces and their refills use is compiled before the first capture (a program first compiled
     later allocates a program-cache buffer, as `_capture`'s note records).
- **Three things today break it:**
  - the CFM's `_capture` allocates its output (`next_x`) inside the capture, and the class keeps one trace at a
    time;
  - the LLM captures its decode trace at every `generate()` and releases it at the end. Its inputs are allocated per
    call and its logits inside the capture;
  - `inference_streaming` keeps its token ids, embeddings and encoder projection alive through the CFM call (found by
    the prototype, below).

## The design

At start-up, after `warmup_buckets()` and `warmup_streaming()`:

1. **Allocate:**
   - every streaming flow bucket's CFM slot (17 buckets), holding the conditioning, `x`, the step's output, the time
     embedding, `dt` and the chunk-causal bias;
   - the LLM decode trace's inputs (token, position, rope index) and its logits.
2. **Compile:** each slot's body twice eagerly, every refill copy (the conditioning, `x`, the time embedding, `dt`,
   output → `x`), and one LLM decode step.
3. **Capture the 17 CFM traces, then the LLM decode trace** (the user's order).
   - The CFM body ends in `ttnn.add(x, step, output_tensor=next_x)`.
   - The LLM head in `ttnn.linear(..., optional_output_tensor=logits)`.
4. **Keep all 18 for the process's life.**
   - `generate()` stops capturing and releasing its own trace.
   - A streamed chunk's CFM takes `_reuse_trace`'s path: copy the conditioning into its bucket's slot, then 10
     replays.
   - The final chunk stays eager: its mask is padded and non-streaming, which the traced solve does not take.
   - A non-streaming request uses the same decode trace, and saves its per-call capture.

What the code would need:
- **`TtCausalConditionalCFM`:**
  - several traces (the capacity is refused above 1 today, for the reason above);
  - `capture_streaming(buckets)` at start-up;
  - outputs into pre-allocated buffers.
- **`TtQwen2LM`:** a start-up capture, kept alive; pre-allocated logits.
- **`CosyVoice2TTNN.warmup_streaming()`:** ends with the captures.
- **The device:** opened with a trace region sized from the measurement below.

## The evidence

The prototype (`scripts/2026-09-30/traced_cfm_prototype.py`) builds the pipeline, runs both warm-ups, streams the
six corpus utterances eagerly as the baseline, and then does 1–3 of the design above. It then streams them again,
each streamed chunk's CFM replaying its bucket's trace and the LLM decoding through its start-up trace. Each run is
one process on the N150, after lever (a). Four things came out on the way; the design above already includes them.

1. **One bucket cannot be traced: 5,120 frames (2,560 tokens).**
   - There, the conv checks chose the raw weight for `down_resnet.block1.conv` (`Conv1d(320->256, k=3)`).
     That is #36487: neither prepared weight was right at that length.
   - The raw weight is a host tensor, so `conv1d` uploads and prepares it on every call. The first capture
     at that size failed with "Writes are not supported during trace capture".
   - So the set of traceable buckets comes from the conv verdicts after the warm-ups, and the rest stay eager: 16 of
     17 buckets here.
   - A chunk meets that bucket only when prompt and tokens reach 2,048.
2. **Three buffers were alive at the first CFM replay.** With the tracker's tracebacks, they are
   `inference_streaming`'s own locals: the token ids, their embeddings and the encoder's projection
   (`tt/flow/flow.py`). None is read after the CFM starts, so this was the tracker's conservatism rather than a
   corruption. Freeing them before the CFM call clears it, and `inference` would get the same change.
3. **A pre-allocated output works in a trace.** On one bucket (512 frames), the 10-step solve matches the eager
   one at PCC 0.999934 in all three forms (`traced_cfm_diag2.py`):
   - `_capture`'s own, the output allocated inside the capture;
   - `ttnn.add(..., output_tensor=)` into a pre-allocated buffer;
   - an in-place `ttnn.add_` into `x`.

   The CosyVoice1 note about an in-trace `ttnn.copy` that never landed does not apply to these.
4. **Compare the audio by log-mel, not by waveform PCC.**
   - The traced step blends and updates on the device in bf16, the eager one on the host in fp32.
   - Call for call, the CFM's input is bit-identical between the two runs, and its output is within PCC
     0.99989–0.99993 (`--record`).
   - Through HiFT's F0-driven sine phase, that 1e-4 difference drifts the waveform: sample PCC falls to about 0 while
     log-mel L1 stays small. The stage A gate judges its own-F0 path by log-mel for the same reason.

**The proof** (`scripts/2026-09-30/phase_traced_cfm3.sh`, run A). The allocation tracker was on
(`TT_METAL_TRACE_ALLOC_TRACKING=1`), with 16 CFM traces and the LLM decode trace alive from start-up.
- **All six utterances streamed with no tracker failure.** That is 17 traced chunk solves (170 CFM replays) and the
  LLM's 1,249 decode replays. The six final chunks stayed eager.
- **The tokens equal the eager run's** for every utterance.
- **The audio is within log-mel L1 0.067–0.100 of the eager run's.** For scale, stage A's own-F0 path measures
  0.069–0.088 against upstream.
- The run's timings are not measurements: the tracker's check before every replay is slow.

**The negative control** (run B) allocates the decode trace's logits inside its capture, as `generate()` does today,
after the CFM traces. The tracker fails the first CFM replay on exactly that one buffer. So the tracker does see
the hazard, and A's clean pass is not vacuous.

**The trace region** (`ttnn.get_memory_view(device, BufferType.TRACE)` around each capture):

| trace | size |
|---|---|
| each CFM trace, 128 to 4,096 mel frames | 6.5–7.8 MB |
| 16 CFM traces | 119.9 MB |
| the LLM decode trace | 3.7 MB |
| **all 17** | **123.6 MB** |

- The pipeline opens the device with a 50 MB trace region today; this needs at least ~124 MB.
- The size does not grow with the bucket, because a trace holds the step's ~1,100 commands, not its data.
- Capturing takes 0.09–0.13 s per trace, and compiling the 16 bodies 5.9 s: about 8 s of start-up in all.
- The 16 slots take 155 MB of DRAM, most of it the chunk-causal biases (T² bf16 each).

## First audio and RTF

**Measured, prototype** (run C: the tracker off; the eager pipeline, then the traced one, in one process after lever
(a); the six utterances, RAS seed 1986, noise seed 1):

| | eager | traced CFM, start-up decode trace |
|---|---|---|
| first audio | 1.337–1.443 s | **0.955–1.046 s** |
| streaming RTF, per utterance | 0.731–1.070 | 0.673–0.898 |
| streaming RTF, aggregate | 0.805 | **0.725** |
| the first chunk's CFM | 0.67–0.68 s | 0.32 s |
| a 100-token hop's CFM (768 frames) | 0.69–0.72 s | 0.54 s |
| a final chunk's CFM (eager either way) | 0.68–0.95 s | 0.69–0.95 s |

- **Where first audio goes, traced:** text and LLM (0.37–0.47 s before, a little less without the per-request capture),
  the flow outside the CFM (0.15–0.19 s), the CFM (0.32 s) and HiFT (0.12 s).
- **Even a free CFM would leave 0.64–0.73 s,** against the 500 ms target.
- **Estimated, not measured: traced at 5 Euler steps** (B37's count, 31 ms a step at 512 frames). The first chunk's CFM
  would be ~0.16 s, and first audio ~0.80–0.89 s.
- **The worst streaming RTF (121-127105-0015, 0.898)** is still carried by its eager final chunk (0.69 s of CFM for
  0.52 s of audio, B35).

## Risks and open points

- **Numerical.**
  - The traced step blends the CFG pair and takes the Euler update on the device in bf16; the eager step does both
    on the host in fp32.
  - The audio moves by log-mel L1 0.067–0.100 against eager. That is the size of stage A's own-F0 difference, which
    never moved WER or SIM, but it is not nothing.
  - Before adopting it: WER/SIM over D43's five draws for the traced configuration, and a listen.
- **Traceable buckets differ by build.** A bucket whose conv verdict is the host raw weight cannot be captured (#36487).
  The set is decided at start-up from the verdicts, and every other bucket stays eager: one of 17 here.
- **The final chunk stays eager.** Its mask is padded and non-streaming. Tracing it needs a second trace per bucket
  (with an all-zero chunk term), or a swap of the chunk-bias buffer's contents: 17 more traces, ~120 MB. It is the
  next RTF lever after this.
- **Memory:**
  - a trace region of ~124 MB against the 50 MB the pipeline asks for today;
  - 155 MB of DRAM for the slots, most of it the chunk-causal biases: T² bf16 each, up to 33.5 MB at 4,096 frames.
- **The decode trace lives for the process.** Nothing persistent may be allocated after start-up. The warm-ups
  guarantee that for the bucketed geometries (D40, D44), and a tracker test in the suite would guard it (a
  streamed request under `TT_METAL_TRACE_ALLOC_TRACKING=1`).
- **Allocation order changes:** start-up allocates the slots before the captures. With conv config tensors in DRAM,
  that costs one cold start of the kernel cache (the known issue).
- **Start-up adds ~8 s:** 16 captures and the bodies' compile.

## What building it would take

1. **`tt/flow/flow.py`:** free `ids_dev`, `tok_emb_dev` and the projection before the CFM call, in both
   `inference_streaming` and `inference`.
2. **`TtCausalConditionalCFM`:**
   - more than one trace;
   - `capture_streaming(buckets)`: allocate every slot, compile every body and copy, then capture;
   - the step's output into a pre-allocated buffer (`output_tensor=`, or an in-place `add_` into `x`, which also
     drops the copy after each replay);
   - skip buckets with a raw-weight verdict.
3. **`TtQwen2LM`:**
   - a start-up capture with pre-allocated inputs and logits (`ttnn.linear(..., optional_output_tensor=)`);
   - `generate()` no longer captures or releases.
4. **`CosyVoice2TTNN`:** `warmup_streaming()` ends with (2) and (3); the device's trace region is sized ≥ 124 MB;
   `live_traces()` and its tests learn the persistent traces.
5. **Tests:**
   - the streaming stage B test under the tracker, with the persistent traces;
   - traced against eager on stage A's gates, which already judge HiFT's own-F0 path by log-mel.
