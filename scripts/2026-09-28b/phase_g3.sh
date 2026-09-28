#!/bin/bash
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928b
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
cd /home/user/tt-metal
start_job repro_conv1d_v bash -c "source python_env/bin/activate && python $SCR/repro_prepare_conv1d.py"
wait_job repro_conv1d_v 3600
grep -E '^\{' $RUN_DIR/repro_conv1d_v.log
start_job hift_single_ctx bash -c "source python_env/bin/activate && python $SCR/hift_single_context.py"
wait_job hift_single_ctx 5400
grep -E '^\{' $RUN_DIR/hift_single_ctx.log
