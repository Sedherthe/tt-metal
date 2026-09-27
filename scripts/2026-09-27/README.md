# 2026-09-27 one-off checks (not part of the PR)

| script | device | answers |
|---|---|---|
| `trailing_silence_seq_lengths.py` | no | R7/R8: were the 09-23 clips near `max_seq_len=512`? (No: 286/350/376/493; clip 1 far from it) |
| `encoder_final_chunk_realweights.py` | yes | R5: real-weight region numbers used to set the final-chunk gates (fixed 0.016-0.067 vs chunk-only control 0.23-0.39) |
| `sdpa_sweep.py` (+ `sdpa_sweep.log`) | yes | R6: real-weight estimator at T = 1 mod 32, fused SDPA vs explicit chain vs torch |
| `sdpa_op_probe.py`, `sdpa_plant_verify.py` | yes | R6: #57608 not reproducible on Wormhole (planted padding verified, output bit-identical) |
| `hift_torch_f0_injection.py` (+ log) | yes | R9: own-F0 vs torch-F0-injected HiFT, real weights, fp32 |
| `ref_mask_check.py` (+ log) | no (reference venv) | Upstream's LLM decode under transformers 5.12.1: its length-1 decode mask is right-padded with zeros, so each step attends to position 0 only (max \|d log p\| 16.4 vs a no-cache forward; 3.1e-5 with the mask over cache + input). The fix is shim 4 in `scripts/reference_env.py`, commit `0d687d840e` |
| `text_goldens.py` (+ `text_goldens.json`) | no (reference venv) | Upstream `text_normalize` + `CosyVoice2Tokenizer.encode` on eight hand-picked inputs: the goldens embedded in `tests/e2e/test_text.py` (`d651a5edfc`) |
| `f0_dtype_rtf_check.py` (+ log) | yes | (f): fp32 vs bf16 F0/source, one process, a shared fp32 decoder. Warm HiFT time is equal within 4 ms (RTF difference <= 0.0011). |

Paths in these scripts are those of 09-27: the model package was `models.demos.audio.cosyvoice2` until it moved to
`models/experimental/cosyvoice2`.
