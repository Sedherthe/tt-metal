#!/bin/bash
# Stage 3, step 2: one CFM Euler step. First wall times (eager, split into parts; traced), then the device profiler
# over one eager step at the first chunk's geometry, then the ops report summarized.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
export HF_HOME=/home/user/models
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
cd /home/user/tt-metal
start_job cfm_time bash -c "echo \$\$ > $RUN_DIR/cfm_time.pid; exec /opt/venv/bin/python $S/cfm_step_profile.py --out $RUN_DIR/cfm_time.json --repeats 5"
wait_job cfm_time 86400 || exit 1
grep -E '^\{"(stream|final)' $RUN_DIR/cfm_time.log
start_job cfm_tracy bash -c "echo \$\$ > $RUN_DIR/cfm_tracy.pid; exec /opt/venv/bin/python -m tracy -r -p -v -o $RUN_DIR/cfm_tracy --op-support-count 5000 $S/cfm_step_profile.py --profile stream_256 --out $RUN_DIR/cfm_profile.json"
wait_job cfm_tracy 86400  # exits 1: see below
# tracy's own ops report fails here (2026-09-30): the compile solve overflows the profiler's DRAM buffers before
# the script's first flush, and the post-processing asserts on those ops. The step itself runs after the flush,
# so its ops are read from the raw logs instead.
python3 $S/cfm_profile_raw.py $RUN_DIR/cfm_tracy/.logs --out $RUN_DIR/cfm_profile_raw.json
