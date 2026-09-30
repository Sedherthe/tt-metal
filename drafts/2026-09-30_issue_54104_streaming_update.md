<!-- DRAFT, not posted. A comment on tenstorrent/tt-metal#54104. Figures: PR #56651 at d2a2439e05 (docs/VALIDATION.md,
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

**Where the time goes.** The first chunk spends:
- 0.37–0.47 s on text and LLM until its tokens are in;
- 0.82–0.91 s on the flow, of which the CFM's 10 Euler steps (with classifier-free guidance) take 0.68–0.74 s;
- 0.12–0.13 s on the vocoder.

Even a free flow would leave first audio at about 0.5–0.6 s. Every chunk reruns the flow over the whole prefix, as upstream does, so each costs at least ~0.8 s. Only one utterance streams at RTF above 1: a 3.8 s sentence whose last 13 tokens need a third chunk, and with it a full non-streaming flow, for 0.5 s of audio.

**Fixed along the way.** The vocoder's padded calls silenced the last ~25 ms of every streamed utterance. On one of them, Whisper heard a trailing "you" in 5 of 11 noise draws. The padded calls are now masked so they compute upstream's call at the real length, and that utterance now transcribes cleanly in all 11 draws.

**Next:** profile one CFM Euler step on device, then sweep 5, 6, 8 and 10 Euler steps, scoring WER and similarity over five noise draws.

**A question on Stage 3.** Fewer Euler steps is the one lever that reaches the 500 ms target, and it changes the output. If the sweep shows WER and similarity holding within noise at fewer steps, would a reduced step count be acceptable for Stage 3? If not, would a measured report with the targets missed be acceptable?
