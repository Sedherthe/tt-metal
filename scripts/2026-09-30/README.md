# 2026-09-30: the "you" clip (B28; not part of the PR)

Stage A's streaming WER was 0.68 % on 09-28 and 1.36 % on 09-29: Whisper appends "you" to 260-123440-0010. The user
asked where the rebuild differs before R6, with no PR code changed until that is known. Nothing here changes the PR:
the device experiments swap one constant or one class in their own process. Paths are this pod's: data under
`/home/user/data`, run output under `/home/user/data/cosyvoice2_runs/0930`. Job control is `../2026-09-28/jobs.sh`.
The spec-against-today table is `../../reviews/2026-09-30_rebuild_spec_vs_today.md`.

| file | device | what |
|---|---|---|
| `you_asr.py`, `phase_you_asr.sh` | no (reference venv) | The PR's scorer call on the clip (ours live, ours offline, upstream streamed, our Stage 1), recording every greedy step's log-probability and runners-up. Five runs as the scorer runs, five seeded, then 24 and 1 threads; the first also transcribes tail splices. |
| `you_asr.jsonl`, `you_asr_summary.py` (+ `you_asr_summary.md`) | no | Deterministic: one transcript and the same per-step log-probabilities, to three decimals, in all 12 runs of each clip. Ours ends in "you" 12 of 12, upstream and our Stage 1 0 of 12. The decision is the first text token, " how" against " How" (ours 0.13 nats lowercase; upstream 0.54 capitalized). Splices: ours with upstream's last 0.1 s is clean; upstream with our last 0.1 s says "you". |
| `you_lengths_tails.py` (+ `.log`) | no | Lengths, whole and per chunk, for all six: identical to upstream's. Tails: the same envelope to 1–2 dB per 20 ms frame, except the last frame, where ours collapses. Writes `~/listening/tails/`. |
| `you_tail_ab.py`, `phase_tail_ab.sh` (+ `.log`, `tail_ab_summary.json`) | yes | The final HiFT call end-padded (as built) against its exact length, over the six cases (R5's noise), ten noise realizations of the clip, and the mechanism (upstream's mel, F0 and noise). The padded variant reproduces R5's and R3's wavs sample for sample. Exact: corpus 0.68 % / 95.88. It compiled 4,046 kernels for five new final lengths. |
| `you_tail_front.py`, `phase_tail_front.sh` (+ `.log`, `tail_front_summary.json`) | yes | A fix candidate: the final call padded in front. The tails are fixed and "you" goes (0 of 11; corpus 0.68 %), but final seams fall to 0.9923–0.9997 (five of six under the gate's 0.998) and SIM to 95.80. Rejected. 0 kernels compiled. |
| `you_mech_metrics.py` (+ `you_mech_metrics.md`) | no | The mechanism outputs of all three variants against upstream: seam, chunk body, tail, last 20 ms. |
| `you_score_sweep.py`, `you_sweep_scores.jsonl`, `front_sweep_scores.jsonl` | no (reference venv) | Whisper on the sweeps and the mechanism outputs. Padded: "you" 5 of 11, first token within ±0.16 nats of a tie; exact: 0 of 11, 0.78–1.04 nats clear. Mechanism: padded says "you" (0.55 nats lowercase), exact doesn't (0.575 capitalized, upstream 0.541). |
| `score_padded.log`, `score_exact.log`, `score_front.log` | no (reference venv) | `eval_wer_sim.py` on the three corpus runs against upstream's streaming. |
