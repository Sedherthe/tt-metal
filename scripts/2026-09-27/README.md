# 2026-09-27 one-off checks (not part of the PR)

| script | device | answers |
|---|---|---|
| `trailing_silence_seq_lengths.py` | no | R7/R8: were the 09-23 clips near `max_seq_len=512`? (No: 286/350/376/493; clip 1 far from it) |
| `encoder_final_chunk_realweights.py` | yes | R5: real-weight region numbers used to set the final-chunk gates (fixed 0.016-0.067 vs chunk-only control 0.23-0.39) |
| `sdpa_sweep.py` (+ `sdpa_sweep.log`) | yes | R6: real-weight estimator at T = 1 mod 32, fused SDPA vs explicit chain vs torch |
| `sdpa_op_probe.py`, `sdpa_plant_verify.py` | yes | R6: #57608 not reproducible on Wormhole (planted padding verified, output bit-identical) |
| `hift_torch_f0_injection.py` (+ log) | yes | R9: own-F0 vs torch-F0-injected HiFT, real weights, fp32 |
