# 2026-09-28 one-off checks and job chains (not part of the PR)

The scripts use absolute paths from the session that ran them: the scratchpad under `/tmp/claude-1000/...` and
`/home/user/data/cosyvoice2_runs/0928`. The package is `models/experimental/cosyvoice2`.

| file | device | answers |
|---|---|---|
| `jobs.sh` | — | D25/B14: `start_job` writes `<name>.exit` with the exit code as its last action. `wait_job` waits on that file; its timeout reports and never kills. |
| `token_precision_experiment.py` (+ `token_precision_arms.txt`, in run order; the first three compiled their head kernels, so their timing is not warm) | yes | B12: teacher-forced token accuracy by LLM head. The arms are default (bf16 logits), fp32out (bf16 weights, fp32 accumulation and logits; adopted), fp32head (fp32 weights), fp32head_hifi3 and bf16. Each arm ran twice for warm timing. |
| `noise_floor.log` | no (reference venv) | B12: `scripts/token_accuracy_reference.py --precision bf16 / bf16-fp32-head --against <fp32 run>` |
| `warmup_measure.py` (+ `warm_dram-cold.json`, `warm_dram-second.json`, `warm_l1.json`) | yes | B15: the bucketed start-up. For each geometry: time, conv safety-check time, weight-prep time and memory. Also free DRAM at every conv-cache insert, and binaries compiled. `--l1` is the bounded L1 option. The cold run's per-geometry binary counts read 0 because of a glob bug (fixed before the second run); its total, 19,068, is a count of the cache directory. |
| `listening_pairs.py` (+ `listening_pairs.json`) | yes | B11: the same tokens, mel and sine noise, vocoded at the HiFT bucket and at the exact length. The wavs are in `~/listening`. `reach_ms` there is swamped by the whole-utterance geometry difference; use the dBFS table in `~/listening/README.txt` instead. |
| `phase_*.sh` | yes | The job chains, in order: A (fp32 head + noise floor), A2 (head variants, warm timing), A3 (the token test through the pipeline), B (cold + second warm-up), C (device suite), D (bounded L1), E1 (Stage 1 demo, cold first request, listening pairs + WER/SIM), E2 (the pytest perf test). |
