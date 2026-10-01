# 2026-10-01: the new pod checked, the environment and reference side rebuilt, the backup tip re-verified (not part of the PR)

The 09-30 pod ended overnight and took `/home/user/data` with it (B42). These chains check the new pod's card, rebuild
the environment and the reference side, and re-verify the backup tip (`213afe7909`) on this card. Paths are this
pod's: data under `/home/user/data`, logs under `/home/user/data/cosyvoice2_runs/1001`. Job control is
`../2026-09-28/jobs.sh` (sentinel files, D25).

| file | device | what |
|---|---|---|
| `phase_card.sh` (+ `tt_smi1.json`, `tt_smi2.json`, `smoke_add.log`; after B43's kill, `tt_smi3_after_kill.json`, `smoke_add_after_kill.log`) | yes | RUNBOOK §1: `tt-smi -s`, `../2026-09-29/smoke_add.py`, `tt-smi -s` 150 s later. n150 L at `0000:e1:00.0`, firmware 19.11.0.0, DRAM OK, `FAULTS` 0; heartbeat 295,904 → 296,201; the smoke test within one bf16 ulp. |
| `phase_env.sh` (+ `phase_env.log`, `phase_env_jobs.log`) | no | In parallel: `inflect` 7.5.0 into `/opt/venv`; the reference venv (`docs/security.md`'s two steps, step 2 locked); upstream at `074ca6dc9e80`, recursive; the checkpoint at the pinned revision only, plus `Qwen/Qwen2-0.5B-Instruct` for the LLM tests; LibriSpeech test-clean, md5-checked. The venv's freeze is byte-identical to `../2026-09-29/ref_env_freeze.txt`. |
| `phase_ref.sh` | no (reference venv) | `../2026-09-29/phase_ref.sh` with today's run directory: the shim self-test, the inputs, both reference runs, the token-accuracy references, the seam gate's reference, the reference's scores. |
| `phase_device.sh` | yes, and the reference venv | The Stage 1 demo once the inputs exist (it fills the empty kernel cache; the reference chain shares the CPU, so its timings are not measurements). Then upstream's streaming of its tokens (stage A's reference), the device suite with every reference, the demo's WER/SIM, and the two perf tests, each in its own process. |
| `detach.sh` | | Starts a chain with `setsid`, out of the harness's reach, writing `NAME.exit` as its last action (B43). |
| `run_all.sh` (superseded by `detach.sh`) | | Its harness command was stopped at 30 minutes, and the device chain with it (B43). The restart: `RUN_DIR=... detach.sh phase_device phase_device.sh`. |
| `phase_recheck.sh` | yes | B44: the op probe, `tt-smi -s`, the token-accuracy test alone, the demo again (the record's tokens), the API test alone (6 passed). |
| `determinism_probe.py` (+ `determinism_probe.json`) | yes | B44: four ops, 200 repeats each on fixed inputs, compared bit for bit. All identical. |
| `llm_determinism_probe.py` (+ `llm_determinism_probe.json`) | yes | B44: `teacher_forced_topk` on one case, 8 repeats traced and 8 eager. Every repeat differs from the first in its decode steps; the prefill never does. |
| `phase_llm_probe.sh` | yes | B44: the op probe again, the LLM probe, then the token-accuracy test twice (95.902 % and 95.942 %). |
| `b44_runs.log`, `tt_smi5_end.json` | yes | B44's evidence from the pod's logs: the diverged demo and the suite's failures, the clean perf tests, demo and API test, the four token-accuracy runs; the card at 13:43 (DRAM OK, no faults, AICLK 500 idle). |
