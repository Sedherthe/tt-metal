#!/bin/bash
# Token-accuracy extension, CPU only (reference venv): inputs, reference runs, teacher-forced top-5, bf16 noise floor.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928b
S=/home/user/tt-metal/models/experimental/cosyvoice2/scripts
PY="env OMP_NUM_THREADS=48 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
IN=/home/user/data/cosyvoice2_inputs
REF=/home/user/data/cosyvoice2_runs/reference_tfx
TOK=/home/user/data/cosyvoice2_token_accuracy
cd /tmp
start_job tfx_inputs bash -c "$PY $S/prepare_inputs.py --extension --out-dir $IN"
wait_job tfx_inputs 3600 || exit 1
start_job tfx_reference bash -c "$PY $S/run_reference.py --extension --out-dir $REF"
wait_job tfx_reference 14400 || exit 1
start_job tfx_token_ref bash -c "$PY $S/token_accuracy_reference.py --inputs $IN --run-dir $REF --out-dir $TOK"
wait_job tfx_token_ref 3600 || exit 1
start_job tfx_noise_floor bash -c "$PY $S/token_accuracy_reference.py --inputs $IN --run-dir $REF --against $TOK --precision bf16 --out-dir $RUN_DIR/token_ref_bf16_tfx && $PY $S/token_accuracy_reference.py --inputs $IN --run-dir $REF --against $TOK --precision bf16-fp32-head --out-dir $RUN_DIR/token_ref_bf16_fp32head_tfx"
wait_job tfx_noise_floor 7200
echo "tfx chain done"
