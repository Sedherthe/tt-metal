#!/bin/bash
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928b
PY="env OMP_NUM_THREADS=32 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
cd /tmp
start_job hift_stream_ref bash -c "$PY /home/user/tt-metal/models/experimental/cosyvoice2/scripts/hift_streaming_reference.py --out-dir /home/user/data/cosyvoice2_hift_stream_ref"
wait_job hift_stream_ref 5400
grep -E "^  [0-9]|Error|Traceback" $RUN_DIR/hift_stream_ref.log | tail -8
