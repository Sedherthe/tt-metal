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
| `watch_chains.sh`, `watch_chain.sh` | no | Emit a chain's job lines as they land, plus a progress line every 10 minutes while a device job runs, so a hang doesn't pass as silence. |
| `pin_check.sh` (+ `.log`) | host only | D35. The pinned revision resolves the same snapshot on both sides, and `prepare_inputs.py` under the pin rewrites all seven cases bit-identically. |
| `osv_audit_lock.log` | no | D35. `../2026-09-27/osv_audit.py` on the reference venv's lock file: the same two advisories `docs/security.md` dispositions, nothing else. |
| `r1_repro_36487.py` (+ `.log`) | yes | B22, B23. #36487's own reproducer, with prepare declaring TILE (as written: PCC 0.000768), declaring ROW_MAJOR (0.999912), and the raw weight (0.999912). |
| `r1_prepare_layout.py` (+ `.log`) | yes | B23. 36 conv1d geometries, TILE declared vs ROW_MAJOR declared vs raw, against a float64 conv. `phase_r1probe.sh` runs both scripts. |
| `phase_r1verify.sh` | yes | R1. The Stage 1 demo with the ROW_MAJOR candidate, bit-compared with `phase_baseline.sh`'s run; then the suite, with R2's reference running on CPU. |
| `r2_select_seam_mels.py` (+ `.log`) | no (reference venv) | R2. Six mels from six speakers, cut so every crossfade is voiced by upstream's own F0 (> 10 Hz, ±4 frames): nine seams. |
| `r2_hift_stream_ref.log` | no (reference venv) | R2. `hift_streaming_reference.py` on those six: the stitch is exact. Upstream's own two calls disagree by 0.13–0.63 over every crossfade. |
| `phase_r23.sh` | yes | R2's seam gate on the new reference, then R3's streaming gate (stage A), then WER/SIM on the offline-streamed audio. Both gates' first measurement set their thresholds (D38, D39). |
| `phase_r23b.sh` | yes | The two gates re-run with those thresholds (both pass), then the encoder and flow tests (56 passed). |
| `r2_seam_gate.log` | yes | B24. The nine-seam table: mechanism 0.041–0.078 over the crossfade, control 0.108–0.473. |
| `r3_streaming_ref.log` | no (reference venv) | B25. `streaming_reference.py` on TT's Stage 1 tokens: each case's chunk plan and HiFT call lengths. |
| `r3_stream_gate.log` | yes | B25. The stage A gate, per chunk: flow, control, HiFT chunk and seam PCC, and the final chunks' tail levels. |
| `r3_scores.log` | no (reference venv) | B25. WER/SIM: upstream streaming (0.68 % / 95.90) and our offline streaming (1.36 % / 95.83) of the same tokens. |
| `phase_r4.sh` (+ `.log`) | yes | B26, D36. R4's hang check first (the opt-in tracker on the CFM traces; then the interleaved test under `TT_METAL_TRACE_ALLOC_TRACKING=1`, alone), then the full suite (210 passed, 3 skipped). |
| `phase_r5.sh` (+ `.log`) | yes | B27. R5: the streaming demo twice (warm; 0 kernels compiled each), the cold first request under the tracker (failed: 416 compiled, then the tracker's refusal), WER/SIM. It waited for `r4_committed.exit`, written by hand after R4's push, so no pre-commit stash could race a demo's imports. |
| `r5_warm1_timings.md`, `r5_warm2_timings.md` | yes | B27. The two warm runs' tables: per utterance, the first chunk's breakdown, first audio and RTF. |
| `r5_cold_excerpt.log` | yes | B27. The cold run's traceback: 1,259 buffers alive at the first decode replay after the first chunk, counted by what allocated them (772 `ttnn.to_device`, 487 program-cache). The full log stays on the pod. |
| `r5_score.log` | no (reference venv) | B27. WER/SIM of run 1 against upstream's streaming of the same tokens: 1.36 % / 95.85 against 0.68 % / 95.90. |
| `r5_tt_smi.json` | yes | `tt-smi -s` after the cold run (15:05): n150 L at `0000:01:00.0`, DRAM OK, heartbeat 158,131, firmware 19.11.0.0. |
| `phase_r5b.sh` (+ `.log`) | yes | D40. The guarded interleaved test, alone, with the schedule test: 2 passed, 0 kernels compiled; the refusal is logged before any device work. |
