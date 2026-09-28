#!/bin/bash
# Phase B: the bucketed start-up, measured. A cold process on an empty kernel cache, then an identical second one.
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
export TT_METAL_CACHE=$RUN_DIR/kcache
cd /home/user/tt-metal
[ -e $TT_METAL_CACHE ] && { echo "$TT_METAL_CACHE exists; the cold run needs an empty cache"; exit 2; }
mkdir -p $TT_METAL_CACHE
for label in dram-cold dram-second; do
  start_job warm_$label bash -c "source python_env/bin/activate && python $SCR/warmup_measure.py --label $label --out $RUN_DIR/warm_$label.json"
  wait_job warm_$label 10800 || { echo "stopping the chain at warm_$label"; exit 1; }
  grep -E '^\{"label' $RUN_DIR/warm_$label.log | cut -c1-1500
done
echo "phase B done"
