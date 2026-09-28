#!/bin/bash
# Head variants, warm: timing comparable across arms (default re-run as the baseline).
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
cd /home/user/tt-metal
for arm in fp32out fp32head_hifi3 default fp32head fp32out fp32head_hifi3; do
  n=token_${arm}_warm; [ -f $RUN_DIR/$n.exit ] && n=${n}2
  start_job $n bash -c "source python_env/bin/activate && python $SCR/token_precision_experiment.py $arm"
  wait_job $n 1800 || { echo "stopping the chain at $n"; exit 1; }
  grep '"arm"' $RUN_DIR/$n.log
done
