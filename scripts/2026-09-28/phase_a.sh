#!/bin/bash
# Phase A: (device) fp32 output-head token accuracy; (CPU, in parallel) the bf16 noise floor, two arms.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
cd /home/user/tt-metal
REF_ENV="env OMP_NUM_THREADS=32 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models /home/user/cosyvoice2_ref_env/bin/python"
REF_ARGS="--inputs /home/user/data/cosyvoice2_inputs --run-dir /home/user/data/cosyvoice2_runs/reference --against /home/user/data/cosyvoice2_token_accuracy"
S=models/experimental/cosyvoice2/scripts/token_accuracy_reference.py

start_job token_fp32head bash -c "source python_env/bin/activate && python $SCR/token_precision_experiment.py fp32head"
start_job noise_floor bash -c "cd /tmp && $REF_ENV /home/user/tt-metal/$S $REF_ARGS --precision bf16 --out-dir $RUN_DIR/token_ref_bf16 && $REF_ENV /home/user/tt-metal/$S $REF_ARGS --precision bf16-fp32-head --out-dir $RUN_DIR/token_ref_bf16_fp32head"
wait_job token_fp32head 2700; a=$?
wait_job noise_floor 5400; b=$?
echo "phase A done: token_fp32head=$a noise_floor=$b"
