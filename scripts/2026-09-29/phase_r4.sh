#!/bin/bash
# R4, with the hang check first (D36): the opt-in allocation tracker on the CFM traces, alone; then the interleaved
# streaming test itself under TT_METAL_TRACE_ALLOC_TRACKING=1, alone; only then the full suite. Each stage stops the
# chain on failure. If the card drops, nothing here resets it.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
T=models/experimental/cosyvoice2/tests
PERF="$T/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances[device_params0]"
cd /home/user/tt-metal
start_job r4_tracker bash -c "echo \$\$ > $RUN_DIR/r4_tracker.pid; exec env COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1 /opt/venv/bin/python -m pytest $T/pcc/test_flow_decoder.py::test_device_cfm_traces_pass_allocation_tracker --timeout=0 -p no:cacheprovider -q -s"
wait_job r4_tracker 86400 || { echo "the tracker baseline failed: stopping before any interleaved run"; exit 1; }
grep -E "tracker subprocess|passed|failed" $RUN_DIR/r4_tracker.log | tail -3
start_job r4_interleaved bash -c "echo \$\$ > $RUN_DIR/r4_interleaved.pid; exec env TT_METAL_TRACE_ALLOC_TRACKING=1 /opt/venv/bin/python -m pytest $T/e2e/test_streaming.py::test_device_streaming_interleaved_with_llm --timeout=0 -p no:cacheprovider -q -s"
wait_job r4_interleaved 86400 || { echo "the interleaved run under the tracker failed: stopping"; grep -E '^E |FATAL|Error' $RUN_DIR/r4_interleaved.log | head -8; exit 1; }
grep -E "chunks|chunk offset|passed|failed" $RUN_DIR/r4_interleaved.log | tail -12
start_job suite_r4 bash -c "echo \$\$ > $RUN_DIR/suite_r4.pid; exec env COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref /opt/venv/bin/python -m pytest $T --timeout=0 -p no:cacheprovider -q -rs --durations=10 --deselect '$PERF'"
wait_job suite_r4 86400
grep -E "passed|failed|^E " $RUN_DIR/suite_r4.log | tail -6
