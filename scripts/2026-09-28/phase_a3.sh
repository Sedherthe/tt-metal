#!/bin/bash
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
cd /home/user/tt-metal
start_job token_accuracy_test bash -c "source python_env/bin/activate && HF_HOME=/home/user/models COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy python -m pytest models/experimental/cosyvoice2/tests/e2e/test_token_accuracy.py models/experimental/cosyvoice2/tests/perf/test_gates.py models/experimental/cosyvoice2/tests/e2e/test_pipeline_api.py -k 'not consecutive' --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job token_accuracy_test 1800
grep -E "^\s+\||lever|PASS|MISS|passed|failed|Error" $RUN_DIR/token_accuracy_test.log | head -30
