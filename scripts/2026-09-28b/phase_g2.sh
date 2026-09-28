#!/bin/bash
# Seam gate with thresholds, then the standalone prepare_conv_weights reproducers (#36487 comparison).
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928b
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
cd /home/user/tt-metal
start_job seam_gate_2 bash -c "source python_env/bin/activate && env HF_HOME=/home/user/models COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref python -m pytest models/experimental/cosyvoice2/tests/pcc/test_hift_chunked.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job seam_gate_2 5400
grep -E "^\| |own F0|control failed|passed|failed|AssertionError" $RUN_DIR/seam_gate_2.log | tail -24
start_job repro_conv1d bash -c "source python_env/bin/activate && python $SCR/repro_prepare_conv1d.py"
wait_job repro_conv1d 3600
grep -E '^\{' $RUN_DIR/repro_conv1d.log
start_job repro_36487 bash -c "source python_env/bin/activate && python $SCR/repro_36487.py"
wait_job repro_36487 3600
grep -E "WITH|Error|error" $RUN_DIR/repro_36487.log | grep -v "Werror" | tail -5
