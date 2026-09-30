<!-- DRAFT, not posted. A comment on tenstorrent/tt-metal#54104. Figures: PR #56651 at 33aa3601eb (docs/VALIDATION.md,
PERF.md), N150, 2026-09-30. -->

Streaming update on CosyVoice2 (draft PR #56651, now in `models/experimental/cosyvoice2/`).

**Streaming runs on N150 (Wormhole).** `synthesize_stream` produces audio chunks while the LLM is still generating. It uses upstream's schedule: hops of 25 tokens (plus the prompt's padding), then 50, then 100, with 3 look-ahead tokens; the final chunk runs the flow non-streaming; HiFT keeps upstream's cache and crossfade between chunks. Each chunk's flow and vocoder run between two LLM decode steps, while the decode trace stays alive.

**It matches upstream's own streaming** (`CosyVoice2Model.tts(stream=True)` over the same tokens, six utterances, 23 chunks):
- identical chunk plan;
- each chunk's new mel within relative L2 0.0085–0.0182 of upstream's;
- vocoder chunks at PCC ≥ 0.9992 and seams at ≥ 0.9990 given upstream's mel, F0 and noise;
- the last 20 ms of every utterance within 0.2–0.5 dB of upstream's.

**Quality** (six LibriSpeech test-clean utterances, 147 words, two speakers; Whisper large-v3 and WavLM-base-plus-sv cosine; each figure is the mean over five draws of the vocoder's noise, with the tokens fixed):

| | WER | speaker similarity |
|---|---|---|
| TT streaming | 0.68 % | 0.958 |
| upstream streaming, same tokens | 0.68 % | 0.959 |
| TT non-streaming (Stage 1) | 0.68 % | 0.959 |
| PyTorch reference, non-streaming | 0.68 % | 0.952 |

The corpus is small. What would you consider a representative set?

**Stage 3 targets: both missed, and measured.** These are after start-up, on distinct utterances, over two fresh processes:
- **time to first audio:** 1.31–1.50 s, against the 500 ms target;
- **streaming RTF:** worst 1.10–1.12 and aggregate 0.84–0.85, against the 0.4 target.

A device test holds both figures inside recorded bands, so they can't drift unnoticed. Stage 1 is unchanged: worst non-streaming RTF 0.654, token accuracy 95.94 %.

**Where the time goes.**
- **The first chunk:** 0.37–0.47 s of text and LLM, then its flow at 0.82–0.91 s, then 0.12–0.13 s of vocoder. The CFM's 10 Euler steps (with classifier-free guidance) take 0.68–0.74 s of that flow.
- **Every chunk reruns the flow** over the prompt and the whole prefix, as upstream does, so no chunk's flow costs less than about 0.8 s. The flows alone take 0.47–0.74 of every utterance's duration.
- **One Euler step at the first chunk's size** takes 65 ms and is host-bound: 62.6 ms of it is the host enqueueing about 1,150 ops, and the same step traced takes 49 ms on the device.
- **Of that device time, 39 % is merging attention heads** (a transpose and a reshape in each of the 56 transformer blocks).

**Fewer Euler steps** (10, 8, 6 and 5, the same five noise draws, TT and upstream):
- **WER and speaker similarity don't move.** WER is 0.68 % at every step count on both sides, and similarity stays within 0.2 of its 10-step value.
- **The audio does change.** At 5 steps it moves from its 10-step version about 1.6–1.9 times as far as our port sits from upstream at 10 steps, and upstream changes as much as we do. So it is the model's own sensitivity; WER and similarity don't measure it.
- **Latency:** each step costs the first chunk about 70 ms. At 5 steps, first audio is 0.98–1.14 s and the worst streaming RTF 0.82–0.84.

So the step count alone reaches neither target.

**A question on Stage 3.** Fewer steps, a cheaper head merge and a traced CFM together still look short of 500 ms. We haven't measured the combination, and the LLM alone takes 0.37–0.47 s before the first chunk can start. Would a measured report be acceptable for Stage 3, with the targets missed, the levers quantified and the step count left at upstream's 10? Or do you want the step count lowered, given that the audio changes while WER and similarity do not?
