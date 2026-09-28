# 2026-09-28, second round: one-off checks and job chains (not part of the PR)

The scripts use absolute paths from the session that ran them: the scratchpad under `/tmp/claude-1000/...` and
`/home/user/data/cosyvoice2_runs/0928b`. They use `jobs.sh` from `scripts/2026-09-28/` (sentinel-file waits).

| file | device | answers |
|---|---|---|
| `select_extension.py` | no | B19: the deterministic selection of the token-accuracy extension (pasted into scripts/corpus.py) |
| `phase_tfx.sh` | no (reference venv) | B19: the extension's inputs, reference runs, teacher-forced top-5, and the bf16 noise floor |
| `prepare_mismatch_probe.py` (+ `.json`) | yes | B17: in real HiFT runs (128, 640, 896 frames), each conv's prepared weight three ways (as the pipeline does, with the conv's compute config, with that and a DRAM slice config) against the raw weight and a float64 host conv |
| `repro_prepare_conv1d.py` (+ log) | yes | B17: the standalone reproducer. Random weights; auto / 2 / 8 DRAM slices, `act_block_h_override=1024`, and an L1 input without slicing |
| `repro_36487.py` (+ log) | yes | B17: #36487's own reproducer on this build (prepared PCC 0.00035, raw 0.999912) |
| `phase_hiftref.sh` | no (reference venv) | B18: `scripts/hift_streaming_reference.py`, upstream's streaming HiFT on the chunked schedule |
| `phase_g1.sh`, `phase_g2.sh` | yes | B19 and B18: token accuracy over 27 sequences, then the seam gate's first measurement and its re-run with thresholds; `phase_g2.sh` also runs the reproducers |
| `hift_single_context.py` (+ log) | yes | B18: TT and upstream single-pass HiFT on the gate's mels. Chunking adds nothing to the port's own spectral error |
| `phase_g3.sh` | yes | B17 and B18: the reproducer variants, then the single-pass context |
| `phase_s.sh` | yes | the device suite on the chunked-HiFT tree (204 passed) |
| `phase_r.sh` (+ `warm_chunked-cold.json`, `warm_chunked-second.json`) | yes | B20: cold and warm start-up on an empty kernel cache, then the Stage 1 demo, the perf test and WER/SIM |
