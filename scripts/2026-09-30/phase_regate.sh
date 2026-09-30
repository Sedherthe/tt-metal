#!/bin/bash
# The Stage 1 and streaming gates on the masked HiFT (5317572d0c): the device suite (the perf test deselected), the
# Stage 1 perf test in its own process, the Stage 1 demo, then the streaming demo twice (R5's protocol). Nothing else
# runs on the host meanwhile, so the timings are measurements.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
export COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
T=models/experimental/cosyvoice2/tests
PERF="$T/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances[device_params0]"
DEMO="/opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS"
elfs() { find $KC -name "*.elf" | wc -l; }
cd /home/user/tt-metal
n0=$(elfs)
start_job suite bash -c "echo \$\$ > $RUN_DIR/suite.pid; exec /opt/venv/bin/python -m pytest $T --timeout=0 -p no:cacheprovider -q -rs --durations=10 --deselect '$PERF'"
wait_job suite 86400
echo "binaries compiled by suite: $(( $(elfs) - n0 ))"
grep -E "passed|failed|^E |SKIPPED" $RUN_DIR/suite.log | cut -c1-220 | tail -12
n0=$(elfs)
start_job perf bash -c "echo \$\$ > $RUN_DIR/perf.pid; exec /opt/venv/bin/python -m pytest '$PERF' --timeout=0 -p no:cacheprovider -q -s"
wait_job perf 86400
echo "binaries compiled by perf: $(( $(elfs) - n0 ))"
grep -E "warmed|^  \| zero|rtf_nonstreaming|passed|failed|^E " $RUN_DIR/perf.log | cut -c1-200 | tail -12
n0=$(elfs)
start_job stage1 bash -c "echo \$\$ > $RUN_DIR/stage1.pid; exec $DEMO --out $RUN_DIR/stage1_masked"
wait_job stage1 86400
echo "binaries compiled by stage1: $(( $(elfs) - n0 ))"
grep -E "^\| zero|Distinct|warmed" $RUN_DIR/stage1.log | tail -8
for run in 1 2; do
  n0=$(elfs)
  start_job stream$run bash -c "echo \$\$ > $RUN_DIR/stream$run.pid; exec $DEMO --stream --out $RUN_DIR/stream_masked$run"
  wait_job stream$run 86400
  echo "binaries compiled by stream$run: $(( $(elfs) - n0 ))"
  grep -E "^\| zero|Distinct|warmed" $RUN_DIR/stream$run.log | tail -9
done
