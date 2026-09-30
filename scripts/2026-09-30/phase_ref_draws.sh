#!/bin/bash
# D42's reference side (CPU, reference venv): five vocoder noise draws, seeds 1-5, over fixed tokens. The PyTorch
# reference's Stage 1 (its own tokens, --seed 1986) and upstream's streaming of TT's Stage 1 tokens.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/tt-metal/models/experimental/cosyvoice2/scripts
PY="env -u PYTHONPATH OMP_NUM_THREADS=32 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
D=/home/user/data/cosyvoice2_draws
cd /tmp
for s in 1 2 3 4 5; do
  start_job ref_stage1_seed$s bash -c "$PY $S/run_reference.py --out-dir $D/ref_stage1_seed$s --noise-seed $s"
  wait_job ref_stage1_seed$s 14400 || exit 1
  start_job ref_stream_seed$s bash -c "$PY $S/streaming_reference.py --inputs /home/user/data/cosyvoice2_inputs --tokens-from /home/user/data/cosyvoice2_runs/0929/stage1_head --out-dir $D/ref_stream_seed$s --noise-seed $s"
  wait_job ref_stream_seed$s 14400 || exit 1
done
echo "reference draws done"
