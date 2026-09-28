#!/bin/bash
# Phase E2: the pytest perf test enforces the recorded rtf_nonstreaming verdict (Stage 1 protocol), on phase B's cache.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
export TT_METAL_CACHE=$RUN_DIR/kcache HF_HOME=/home/user/models COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
cd /home/user/tt-metal
elfs() { find $TT_METAL_CACHE -name "*.elf" | wc -l; }
n0=$(elfs)
start_job stage1_perf_test bash -c "source python_env/bin/activate && python -m pytest models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job stage1_perf_test 10800
n1=$(elfs); echo "binaries compiled by the pytest process: $((n1 - n0))"
grep -E "start-up|^  \||PASS|MISS|lever|passed|failed|Error" $RUN_DIR/stage1_perf_test.log | head -30
