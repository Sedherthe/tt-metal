# `REBUILD_2026-09-29.md`: every measured figure against today's (2026-09-30)

The user asked for this before R6: stage A's streaming WER was 0.68 % on 09-28 and is 1.36 % now, so the rebuild
might differ from the lost code. Every figure the spec states for R1–R5 sits next to the value measured on 09-29
(this pod, KMD 2.9.0), with its log under `scripts/2026-09-29/` unless noted. "Matches" means within the run-to-run
spread of the measurement. The "you" clip itself is B28 and `scripts/2026-09-30/`.

## R1: #36487, the ROW_MAJOR-prepared candidate

Relative error against a float64 conv: declared TILE / declared ROW_MAJOR / raw weight (`r1_prepare_layout.log`,
36 geometries).

| spec row | spec (09-28) | today | verdict |
|---|---|---|---|
| resblock k=11, d=1/3/5, 5,120 | 1.40–1.51 / 0.0039–0.0041 / = RM | 3.78, 1.38, 1.38 / 0.00394–0.00413 / = RM | same verdicts. The TILE error's size differs (3.78): a wrong weight's error depends on the probe's inputs and is not a stable figure |
| source_downs[0] k=30 s=15, 8 lengths | 1.23 / 0.0039–0.0046 / = RM | 1.23 (640–2,048 frames) / 0.00396 / = RM | matches |
| source_downs[1] k=6 s=3, 6 lengths | 1.13–1.15 / 0.0039–0.0057 / = RM | 1.145–1.148 (896–2,048 frames) / 0.00391 / = RM | matches |
| "flow decoder conv k=3 (5,120)": clean under TILE, broken under RM | 0.0029 / 1.33 / 0.0029 | the CFM's 256→256 k=3 at 5,120: 0.00315 / 0.00315 / 0.00315 | **not matched.** The spec doesn't name the conv's channels. The one flow k=3 conv at 5,120 that TILE gets right is right under RM too today |
| resblock k=11 at 10,240, clean | 0.0039 / 1.37 / 0.0039 | 0.00395 / 1.37 / 0.00395 | matches |
| source_downs[0] at 61,441, clean | 0.0018 / 1.3e36 / 0.0018 | 0.00185 / 1.13 / 0.00185 | same verdicts; RM's error size differs, as above |
| the CFM's 320→256 k=3 at 5,120 (2,560-token bucket): broken both ways, raw kept | 2.13 / 2.27 / 0.0026 | 1.26 / 1.36 / 0.00357 | same verdicts. The raw weight's error is 0.0036 against 0.0026: the probe's inputs (not recorded in the spec) set it |
| #36487's own reproducer, PCC TILE → RM | 0.225 → 0.999912 | 0.000768 → 0.999912; raw 0.999912 | TILE doesn't reproduce (B22, D37: today's figure is used); RM matches to the digit |
| unit test on a real broken geometry: TILE / RM / raw | inf / 0.0039 / 0.0039 | inf / 0.0038 / 0.0038 | matches |
| Stage 1 output with R1 | unchanged | bit-identical (`phase_r1verify.sh`) | matches |
| recompiled kernels | ~898 | 898 | matches |
| rejected: conv inputs to ROW_MAJOR (30 convs broken, HiFT 22–28 % slower, PCC 0.9909 → 0.9966) | | not re-measured | |

## R2: the seam gate

| spec figure | spec | today (`r2_seam_gate.log`, `r2_hift_stream_ref.log`) | verdict |
|---|---|---|---|
| mechanism, relative error over the crossfade | 0.049–0.078 | 0.041–0.078 | same bound held. The seams are a different set: the spec doesn't name its six mels, so today's were chosen again (B24) |
| no-crossfade control | 0.116–0.640, fails at all 9 | 0.108–0.473, fails at all 9 | same verdict, different seams |
| upstream's own two calls agree within 0.03–0.10 at 5 of 9 seams (0.19–0.76 at the rest) | | they disagree by 0.13–0.63 at all 9 | **does not reproduce** on today's seams (B24); a property of the seam set |
| the crossfade's gain | ~8 % | halves sum to ~1.08 | matches |
| margin | 0.078 against 0.10 | 0.078 against 0.10 | matches |

## R3: streaming stage A

| spec figure | spec | today (`r3_stream_gate.log`, `r3_scores.log`) | verdict |
|---|---|---|---|
| mel per chunk, relative error | 0.0085–0.018 | 0.0085–0.0182 | matches |
| vocoder seams, PCC | ≥ 0.999 | 0.99900–0.99989. The lowest (121-127105-0024 seam 3) failed a strict 0.999 on the first run by less than 5e-6 | noise-level; the gate is now 0.998 (D38) |
| final chunks' emitted audio | not stated | PCC 0.96492 on 121-127105-0015's 13-token final chunk, tail included (first run) | new here: the end-padded tail. D38 gated it on level instead |
| WER / SIM, ours | 0.68 % / 95.81 | 1.36 % / 95.83 | **flag**: one inserted "you" on 260-123440-0010 (B28) |
| WER / SIM, upstream streaming | 0.68 % / 95.89 | 0.68 % / 95.90 | matches |

## R4: streaming stage B

| spec figure | spec | today (`phase_r4.log`, `r5_guard`) | verdict |
|---|---|---|---|
| hang | none (KMD 2.3.0) | none (KMD 2.9.0), tracked and untracked | matches |
| streamed audio against offline streaming | bit-identical | bit-identical | matches |
| greedy streamed tokens against batch | equal | equal | matches |
| #36487 at 108 frames | six convs, all fixed by R1 | six (Conv1d 128→128 k=11 at 4,320), ROW_MAJOR candidate chosen | matches |

## R5: streaming measured

| spec figure | spec | today (B27) | verdict |
|---|---|---|---|
| time to first audio | 1.34–1.49 s | 1.336–1.479 s | matches |
| streaming RTF | 0.80–1.12, aggregate 0.85 | 0.787–1.122, aggregate 0.843 / 0.853 | matches |
| first chunk: text and LLM (28–35 tokens) | 0.38–0.47 s | 0.371–0.466 s | matches |
| first chunk: flow | 0.79–0.90 s | 0.812–0.920 s | 0.02–0.03 s slower |
| first chunk: CFM | 0.66–0.70 s | 0.674–0.731 s | up to 4 % slower; a second board could explain it, not investigated |
| first chunk: HiFT | 0.12 s | 0.121–0.127 s | matches |
| first audio with a free flow | 0.50–0.59 s | 0.51–0.59 s | matches |
| cold first request, no warm-up | 172 s to first audio, RTF 65 | refused: the allocation tracker found 1,259 buffers allocated under the live trace (B27, D40) | not measurable as specified |
| the prompt's share of the flow | 15–19 % | not re-measured | |
| a CFM Euler step, 128 or 512 frames | ~65 ms either way | 67–73 ms on the first chunk only | not re-measured as a length sweep |

## What differs, and why

Everything the spec measured reproduces, within run-to-run spread or with the same verdict, except:
- the "you" (R3's WER), explained below;
- figures that depend on inputs the spec didn't record: the R1 probe's inputs, and R2's seam set;
- the R1 "flow decoder conv k=3" row, which matches no flow conv today;
- the cold streaming request, which is now refused (B27).

**The "you" is not a rebuild difference in any figure the spec gives, and not the scorer.**
- The scorer is deterministic, and makes the same call as CosyVoice1's.
- Lengths match upstream exactly, whole and per chunk.
- The cause is the final HiFT call's end padding with silence: the spec's own design ("the final chunk [is] padded with silence to 128 or 256"; only the first chunk is singled out for front padding), recorded as D38. HiFT's look-ahead sees the silence, and our last ~25 ms collapse where upstream's audio runs on.
- Run through upstream's own mel, F0 and noise, that padding alone makes Whisper say "you". The same call at its exact length matches upstream's tail to within 1.3 dB and restores the corpus to 0.68 % / 95.88.
- Across noise realizations, the padded clip says "you" 5 of 11 times, so 09-28's 0.68 % was most likely the same padding on the lucky side (B28).
- So the audio didn't match upstream "as closely as on 09-28" in the sense the gates implied. The gates never checked the last few milliseconds: D38's −50 dBFS floor passed this clip's tail at 2.5 dB below its signal.

