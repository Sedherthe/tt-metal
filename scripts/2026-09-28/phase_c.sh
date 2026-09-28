#!/bin/bash
# Phase C: the package's device suite (default kernel cache), for the commits. The perf test runs in phase E2.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
cd /home/user/tt-metal
start_job suite bash -c "source python_env/bin/activate && HF_HOME=/home/user/models COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy python -m pytest models/experimental/cosyvoice2/tests --timeout=0 -p no:cacheprovider -q -rs --durations=15 --deselect 'models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances[device_params0]'"
wait_job suite 14400
grep -E "passed|failed|^FAILED|^ERROR" $RUN_DIR/suite.log | tail -15
