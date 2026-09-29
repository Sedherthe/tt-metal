# 2026-09-29 checks and job chains (not part of the PR)

The pod that expired on 09-28 took the data directories with it. These chains rebuilt the reference side on the new
pod and re-verified HEAD (`7bd094cc3e`) on it: B21. Paths are this pod's: the data lives under `/home/user/data`, the
logs under `/home/user/data/cosyvoice2_runs/0929`. Job control is `../2026-09-28/jobs.sh` (sentinel files, D25).

| file | device | what |
|---|---|---|
| `smoke_add.py` (+ `smoke_add.log`) | yes | RUNBOOK §1 smoke test: a 64×64 bf16 `ttnn.add` read back. Within one bf16 ulp (max \|diff\| 0.0078). |
| `tt_smi.json`, `tt_smi2.json` | yes | `tt-smi -s` at 09:26 and 10:00: n150 L at `0000:01:00.0`, KMD 2.9.0, firmware 19.11.0.0, DRAM OK, heartbeat 117,882 → 121,913. |
| `ref_env_freeze.txt` | no | The rebuilt reference venv (109 packages). Against `../2026-09-27/ref_env_freeze_tf5121.txt`, only three unpinned indirect dependencies differ: msgpack 1.2.3, platformdirs 4.12.1 and regex 2026.9.29. |
| `phase_ref.sh` | no (reference venv) | Shim self-test; `prepare_inputs.py` (primary with parity, then the extension); `run_reference.py` (both sets); `token_accuracy_reference.py` (both sets); `hift_streaming_reference.py`; `eval_wer_sim.py` on the reference run. `PYTHONPATH` is unset (RUNBOOK §2). |
| `phase_suite.sh` | yes | The device suite in `/opt/venv` (the perf test deselected), then the perf test in its own process. It waits for `phase_ref.exit`. |
| `phase_baseline.sh` | yes | The Stage 1 demo on HEAD, the bit-exact baseline for R1, then its WER/SIM scoring once the demo has exited. |
| `watch_chains.sh` | no | Emits the chains' job lines as they land, plus a progress line every 10 minutes while a device job runs, so a hang doesn't pass as silence. |
