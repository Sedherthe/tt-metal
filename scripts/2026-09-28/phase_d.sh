#!/bin/bash
# Phase D: the L1 option, bounded. Config tensors in L1; stop once L1_SMALL passes 160 KiB.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
export TT_METAL_CACHE=$RUN_DIR/kcache
cd /home/user/tt-metal
start_job warm_l1 bash -c "source python_env/bin/activate && python $SCR/warmup_measure.py --label l1-bounded --l1 --l1-small-kib 192 --stop-at-l1-small-kib 160 --out $RUN_DIR/warm_l1.json"
wait_job warm_l1 10800
grep -E '^\{"label' $RUN_DIR/warm_l1.log | cut -c1-1500
