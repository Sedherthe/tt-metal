# `COSYVOICE1_LESSONS.md`: verification (2026-09-27)

The user uploaded the lessons doc on 09-27 and asked for every claim to be checked. Sources checked:
- the merged CosyVoice1 tree (PR #52540, merged head `f8620f739c`, with no later changes on `main`);
- `tenstorrent/tt-metal` `main` (GitHub API);
- PR #52540's review thread;
- issue #54104.

The user corrected the doc on their side the same day, including the location.

## Contradicted

| claim in the doc | what the sources say |
|---|---|
| The code sits in `models/demos/cosyvoice/`, and asking about the location (finding 18) was unnecessary ("Don't ask") | `main` has `models/experimental/cosyvoice` (200); `models/demos/cosyvoice` returns 404. The code was in `models/demos/cosyvoice` during review (codeowner bot, 08-24). On 09-23 mtairum answered the relocation question: "Yes, please do so." CODEOWNERS: `models/experimental/` → @uaydonat @mtairum; `models/demos` → @mtairum @yieldthought @cglagovichTT @uaydonat. Consequence: D19. |
| Two requirements files, `requirements-cosyvoice.txt` and `requirements-reference.txt` | Only `requirements-reference.txt` exists; its header says the device side installs nothing. |
| `docs/VALIDATION.md` carries no numbers | Partly false: 13 of its lines carry decimal figures. |

## Verified

- **Jeremy's asks, verbatim.** Two comments on PR #52540: the 08-11 guidance and the 08-28 pre-payout asks. The
  asks cover a security disposition for three moderate torch advisories; validation evidence for Stage 3
  batching/pipelining and the device matrix; that the tests don't enforce the RTF thresholds; and that streaming
  begins only after token generation.
- **In the CosyVoice1 tree:**
  - `tests/perf/gates.py`'s semantics: `Meets` / `Misses` / `MissesUnrecorded`, bands checked in both directions;
  - the peak value 72.5 (`generator.py:283`);
  - `n_chunks >= 2` (`test_streaming.py:225`);
  - explicit seeds and the concat alias (`test_batching.py`);
  - `demo --stream`;
  - the L1_SMALL notes;
  - the per-architecture batched-decode PCC.
- **The advisory count.** `docs/security.md` dispositions four advisories (three torch MEDIUM, one transformers
  HIGH). That is one more than the three in Jeremy's ask.
- **Goldens are git-ignored.** `tests/golden/.gitignore` lists `*.npz` and `manifest.json` ("deliberately NOT
  committed yet"). This was first reported as not found and corrected the same evening.

## Not verifiable from the merged tree

Neither confirmed nor contradicted:
- **The 161.7 s → 14.8 s kernel-cache effect.** Not in CosyVoice1's committed files. Our own measurement reproduces
  it in kind: 321 s against 14.3 s for the same 7 s utterance, before and after its kernels were on disk.
- **The other figures:** the 65 s TTFP, about 50 KB per seam, the lost hook, the PCC of 0.80, and "N150 never
  measured".

## Beyond the doc

From PR #52540's thread and the issue:
- **Jeremy, 09-14:** finish by 9/30.
- **Jeremy, 09-18, the remaining gaps:** streaming corruption, the Wormhole streaming wedge, batching blocked by
  L1/vocoder growth, n300 RTF 0.552, and quality evidence.
- **mtairum, 09-23 (CI):** n300 RTF 0.553, p150 0.370. A test that asserted a defect's band failed once the defect
  was fixed.
- **mtairum, 09-24:**
  - Wormhole streaming mel PCC was 0.21 because prepared-weight verification was paused at the Wormhole conv1d defect
    geometry (`Conv1d(128->128, k=11)`, L = 8321). He suggested verifying stream geometries during warm-up.
  - SDPA padding fill costs about 8 % of the n300 flow.
- **#54104:** Stage 3 does not list batching, and Stage 1 says "N150 or N300".

## What it changed here

- **The location:** D19.
- **Conv verification:** never paused. Stream geometries get verified in a warm-up (design rule, streaming step).
- **The gates scheme:** adopted in `tests/perf/gates.py` (`0dbe9c44f4`).
- **Memory across consecutive lengths:** an API test runs consecutive different lengths and reports device memory.
  In `0dbe9c44f4`, L1_SMALL stays flat at 0 B/bank.
- **Goldens:** not committed. Prompt inputs come from the directory named by the `COSYVOICE2_INPUTS` environment
  variable, and the tests skip without it.
