# 2026-10-02: reference audio from the merged PR head (not part of the PR)

| file | device | what |
|---|---|---|
| `phase_audio.sh` (+ `phase_audio.log`) | yes, then the reference venv | `33aa3601eb` (the merged head) checked out in `~/tt-metal`: the Stage 1 demo with `--parity`, which synthesized all 27 corpus cases, then the streaming demo. Each run's tokens are checked against the record, and a run that differs is run once more. Then WER/SIM. The first Stage 1 run sampled other tokens (B44) and was discarded; the second and the streaming run sampled the record's. Its token check read the cases by position, which is wrong for a 27-case run. The check by case id, in `../../audio/2026-10-02_merged_33aa3601eb/README.md`, is the one that counts. Output: `../../audio/2026-10-02_merged_33aa3601eb/`. |
