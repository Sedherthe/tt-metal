#!/bin/bash
# 2026-09-29, Step 1: regenerate the corpus inputs and the reference outputs in the rebuilt reference venv (CPU only).
# Same paths and commands as the 09-28 chains (notes: scripts/2026-09-28*/phase_*.sh). PYTHONPATH is unset: this pod's
# ~/.bashrc exports PYTHONPATH=$TT_METAL_HOME, which the reference venv must not see.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
S=/home/user/tt-metal/models/experimental/cosyvoice2/scripts
T=/home/user/tt-metal/models/experimental/cosyvoice2/tests/reference/test_reference_env.py
PY="env -u PYTHONPATH OMP_NUM_THREADS=48 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
IN=/home/user/data/cosyvoice2_inputs
REF=/home/user/data/cosyvoice2_runs/reference
REF_TFX=/home/user/data/cosyvoice2_runs/reference_tfx
TOK=/home/user/data/cosyvoice2_token_accuracy
HREF=/home/user/data/cosyvoice2_hift_stream_ref
cd /tmp
for d in $IN $REF $REF_TFX $TOK $HREF; do [ -e "$d" ] && { echo "$d exists; refusing to mix runs"; exit 2; }; done

start_job ref_selftest bash -c "$PY $T"
wait_job ref_selftest 3600 || exit 1
start_job inputs bash -c "$PY $S/prepare_inputs.py --parity --out-dir $IN"
wait_job inputs 3600 || exit 1
start_job inputs_tfx bash -c "$PY $S/prepare_inputs.py --extension --out-dir $IN"
wait_job inputs_tfx 3600 || exit 1
start_job reference bash -c "$PY $S/run_reference.py --parity --out-dir $REF"
wait_job reference 14400 || exit 1
start_job token_ref bash -c "$PY $S/token_accuracy_reference.py --inputs $IN --run-dir $REF --out-dir $TOK"
wait_job token_ref 3600 || exit 1
start_job reference_tfx bash -c "$PY $S/run_reference.py --extension --out-dir $REF_TFX"
wait_job reference_tfx 14400 || exit 1
start_job token_ref_tfx bash -c "$PY $S/token_accuracy_reference.py --inputs $IN --run-dir $REF_TFX --out-dir $TOK"
wait_job token_ref_tfx 3600 || exit 1
start_job hift_stream_ref bash -c "$PY $S/hift_streaming_reference.py --out-dir $HREF"
wait_job hift_stream_ref 5400 || exit 1
start_job score_reference bash -c "$PY $S/eval_wer_sim.py --run-dir $REF"
wait_job score_reference 7200 || exit 1
echo "reference chain done"
