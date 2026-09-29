#!/bin/bash
# 2026-09-29, Step 1: the package's device suite, then the perf test in its own process (as on 09-28: phase_s + the
# perf step of phase_r). It runs in /opt/venv: this pod has no python_env; inflect 7.5.0 was added with the user's OK.
# It starts only after the whole reference chain has finished (the CPU is free) and its suite inputs exist.
# No timeouts: wait_job only reports. Each job's pid goes to <name>.pid, for SIGINT only.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
export COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
export COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref
PY=/opt/venv/bin/python
T=models/experimental/cosyvoice2/tests
PERF="$T/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances[device_params0]"
cd /home/user/tt-metal

wait_job phase_ref 86400  # the whole reference chain has finished, whatever its code
for step in inputs inputs_tfx token_ref token_ref_tfx hift_stream_ref; do
  [ "$(cat $RUN_DIR/$step.exit 2>/dev/null)" = 0 ] || { echo "reference step $step did not succeed; not starting the suite"; exit 1; }
done

elfs() { find $KC -name "*.elf" | wc -l; }
n0=$(elfs); echo "$(date '+%F %T') binaries on disk before the suite: $n0"
start_job suite bash -c "echo \$\$ > $RUN_DIR/suite.pid; exec $PY -m pytest $T --timeout=0 -p no:cacheprovider -q -rs --durations=15 --deselect '$PERF'"
wait_job suite 86400
n1=$(elfs); echo "$(date '+%F %T') binaries compiled by the suite: $((n1 - n0))"
grep -E "passed|failed|^FAILED|^ERROR|^SKIPPED" $RUN_DIR/suite.log | tail -15

start_job perf bash -c "echo \$\$ > $RUN_DIR/perf.pid; exec $PY -m pytest '$PERF' --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job perf 86400
n2=$(elfs); echo "$(date '+%F %T') binaries compiled by the perf test: $((n2 - n1))"
grep -E "start-up|^  \||PASS|MISS|passed|failed" $RUN_DIR/perf.log | tail -20
