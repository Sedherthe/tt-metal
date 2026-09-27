# CosyVoice2 bring-up — STATUS (2026-09-27, late evening)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). FINDINGS prefixes: R = the 09-27 review, B = build findings (steps 3–5),
O = older open items, X = fixed. Older narrative lives in `history/`.

## Where things stand

- **Bounty #54104:** assigned to Sedherthe. The user posts the issue comments; there's a 2-week reassignment rule
  (R1).
- **PR #56651:** head `bringup/cosyvoice2-istft`, pushed at `0dbe9c44f4`. The user owns the title and description.
- **Built today, design rev 2, steps 1–5** (every commit verified, no Co-Authored-By, pushed by the user):

  | commit | what |
  |---|---|
  | `7d45d9147a` | requirements split, reference venv, `docs/security.md` |
  | `0d687d840e` | corpus (LibriSpeech test-clean, speakers 260 M / 121 F), `prepare_inputs.py`, `run_reference.py`, the reference fixes (B1) |
  | `d651a5edfc` | `tt/text.py`, `tt/prompt.py`, host tests |
  | `0dbe9c44f4` | `tt/pipeline.py`, demo, `test_pipeline_api.py`, `tests/perf/gates.py`, `docs/VALIDATION.md` |

- **The first distinct-utterance numbers** (B2): RTF 21–75 cold, 2.0–2.9 with the kernels on disk, 0.39–0.56 for a
  length the process already ran.
  - Memory across consecutive lengths is flat where it matters (B4).
  - The fp32 F0 default costs nothing (B5).
- **The PyTorch reference** now generates correctly (B1). It runs on CPU at RTF 5.9–7.3.
- **The lessons doc** is verified: `reviews/2026-09-27_cosyvoice1_lessons_verification.md`. The user corrected it.

## Order of work (the user's 09-27 evening decisions)

1. The notes commit (this one).
2. Move to `models/experimental/cosyvoice2/` (D19), its own commit.
3. Evaluation of the existing wavs: WER and similarity, TT against the post-fix reference (D21). No device time.
4. Pin the reference venv's transformers to upstream's version. Remove the patches if the pin makes them unnecessary
   (D20).
5. Non-streaming bucketing, flow and HiFT, with gates (D17). Then the Stage 1 RTF protocol (D18), and record the
   `gates.py` verdict.
6. Streaming (R12, D22), then TTFP and streaming RTF.

## Open questions

- B3: why the disk kernel cache is only partly reused across processes.
