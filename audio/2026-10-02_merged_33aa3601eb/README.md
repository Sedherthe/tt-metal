# CosyVoice2 on TTNN: audio from the merged PR head (2026-10-02)

Audio rendered from `33aa3601eb`, the head of #56651 that was merged into tenstorrent/tt-metal main as `98a19fd851` on
2026-10-01. Main's commit has the same model code on a newer tt-metal. Kept for listening and future reference; not
part of the PR.

## What's here

| folder | what |
|---|---|
| `stage1/` | non-streaming synthesis (`demo.py`): the six corpus utterances and the CosyVoice1-parity sentence |
| `streaming/` | streaming synthesis (`demo.py --stream`): the six corpus utterances, chunk by chunk while the LLM generates |
| `reference/` | the PyTorch reference (upstream CosyVoice2 on CPU, `scripts/run_reference.py`): the same seven texts and prompts |
| `prompts/` | the voice prompts the synthesis clones |
| `cases.json` | per case: the text, the prompt, tokens, audio length, RTF, first audio, WER and SIM, and Stage 1's speech tokens |

All the wavs are 24 kHz, 16-bit mono.

## How it was made

- **The device and code:**
  - an N150 (`0000:e1:00.0`, pod `app-7a544cdc-deployment-65d799ff57-9l4dj`);
  - `33aa3601eb` checked out in `~/tt-metal`, with the build of its tt-metal base;
  - `CosyVoice2Config.reported()`: 10 Euler steps, the fp32-logit LLM head, HiFT in fp32;
  - LLM seed 1986, after `--warmup buckets`.
- **The commands:** `scripts/2026-10-02/phase_audio.sh` in the notes, which runs `demo.py --parity` and then
  `demo.py --stream`. `--parity` synthesized all 27 corpus cases; only these seven are kept.
- **The tokens are the record's:** 213/95/347/75/317/202 for the six corpus utterances, both modes. That is what
  the 09-28 to 09-30 runs sampled, on two other cards.
  - This card's decode is not always reproducible (FINDINGS B44). The first Stage 1 run sampled other tokens
    (202/115/343/78/301/218), so it was discarded, and the second run is the one kept here.
  - The streaming run matched on its first attempt.
- **The reference** was regenerated on 2026-10-01 in the reference venv. It reproduces the recorded reference
  exactly (audio lengths, per-case WER/SIM; FINDINGS B42).
  - It samples its own tokens, so its utterances are a different take of the same text.
- **Scoring:** `scripts/eval_wer_sim.py`, with Whisper large-v3 for WER and `microsoft/wavlm-base-plus-sv`
  x-vector cosine × 100 for SIM, against the prompt utterance.

## Per utterance

RTF and first audio come from one run each on this card, so they are illustrations, not measurements for the
record. PERF.md has those.

| utterance | audio s | tokens | Stage 1 RTF | Stage 1 WER % / SIM | streaming first audio s | streaming RTF | streaming WER % / SIM | reference audio s | reference WER % / SIM |
|---|---|---|---|---|---|---|---|---|---|
| 121-127105-0003 | 8.52 | 213 | 0.436 | 0.00 / 94.73 | 1.464 | 0.800 | 0.00 / 94.44 | 7.64 | 0.00 / 94.32 |
| 121-127105-0015 | 3.80 | 95 | 0.582 | 0.00 / 94.72 | 1.416 | 1.067 | 0.00 / 93.77 | 3.68 | 0.00 / 92.61 |
| 121-127105-0024 | 13.88 | 347 | 0.486 | 0.00 / 93.52 | 1.376 | 0.798 | 0.00 / 93.87 | 12.84 | 0.00 / 93.94 |
| 260-123286-0014 | 3.00 | 75 | 0.620 | 0.00 / 95.95 | 1.380 | 0.932 | 0.00 / 96.34 | 3.12 | 14.29 / 95.82 |
| 260-123440-0002 | 12.68 | 317 | 0.469 | 2.27 / 98.30 | 1.398 | 0.825 | 2.27 / 98.49 | 12.04 | 0.00 / 97.74 |
| 260-123440-0010 | 8.08 | 202 | 0.455 | 0.00 / 98.26 | 1.380 | 0.841 | 0.00 / 98.35 | 8.32 | 0.00 / 96.85 |
| parity_cosyvoice1_en | 6.24 | 156 | 0.451 | 6.67 / 93.91 | — | — | — | 6.04 | 0.00 / 94.86 |

- **The six corpus utterances:**

  | | WER | SIM |
  |---|---|---|
  | Stage 1 | 0.68 % (1 error in 147 words) | 95.91 |
  | streaming | 0.68 % | 95.88 |
  | the reference | 0.68 % | 95.21 |

- **The errors are Whisper's:**
  - 260-123440-0002: "white Rathor" for "white rabbit", in both modes. It is the same error upstream's own
    streaming makes on these tokens (FINDINGS B30).
  - The parity sentence: "That quick brown fox" for "The quick brown fox".

## The texts and prompts

| utterance | text | prompt |
|---|---|---|
| 121-127105-0003 | There was a unanimous groan at this and much reproach after which in his preoccupied way he explained. | `prompts/121-121726-0003.flac` |
| 121-127105-0015 | He quitted the fire and dropped back into his chair. | `prompts/121-121726-0003.flac` |
| 121-127105-0024 | Poor douglas before his death when it was in sight committed to me the manuscript that reached him on the third of these days and that on the same spot with immense effect he began to read to our hushed little circle on the night of the fourth. | `prompts/121-121726-0003.flac` |
| 260-123286-0014 | Truly this sea is of infinite width. | `prompts/260-123286-0016.flac` |
| 260-123440-0002 | It was the white rabbit returning splendidly dressed with a pair of white kid gloves in one hand and a large fan in the other he came trotting along in a great hurry muttering to himself as he came oh the duchess the duchess. | `prompts/260-123286-0016.flac` |
| 260-123440-0010 | How cheerfully he seems to grin how neatly spread his claws and welcome little fishes in with gently smiling jaws. | `prompts/260-123286-0016.flac` |
| parity_cosyvoice1_en | The quick brown fox jumps over the lazy dog while the morning sun rises slowly. | `prompts/zero_shot_prompt.wav` (a Mandarin prompt: cross-lingual cloning) |

## Sources and licenses

- **The prompts** `121-121726-0003.flac` and `260-123286-0016.flac` are from LibriSpeech test-clean (openslr 12;
  V. Panayotov et al.), CC BY 4.0.
- **`zero_shot_prompt.wav`** is upstream CosyVoice's `asset/zero_shot_prompt.wav` (FunAudioLLM/CosyVoice at
  `074ca6dc9e80`, Apache-2.0).
- **The texts** are the LibriSpeech transcripts of the target utterances. The parity sentence is the CosyVoice1
  bring-up's.
- **The synthesized audio** comes from the FunAudioLLM/CosyVoice2-0.5B checkpoint at `eec1ae6c`.
