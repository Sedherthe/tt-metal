#!/bin/bash
# After the prepare probe: token accuracy over all 27 cases (the new code), then the first seam-gate measurement.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928b
wait_job prepare_probe 7200 || true
cd /home/user/tt-metal
ENVS="HF_HOME=/home/user/models COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref"
start_job token_acc_all bash -c "source python_env/bin/activate && env $ENVS python -m pytest models/experimental/cosyvoice2/tests/e2e/test_token_accuracy.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job token_acc_all 3600
grep -E "^  \||^  set |margin|PASS|MISS|passed|failed" $RUN_DIR/token_acc_all.log | tail -40
start_job seam_gate_1 bash -c "source python_env/bin/activate && env $ENVS python -m pytest models/experimental/cosyvoice2/tests/pcc/test_hift_chunked.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job seam_gate_1 5400
grep -E "^\| |own F0|passed|failed|Error|assert" $RUN_DIR/seam_gate_1.log | tail -40
