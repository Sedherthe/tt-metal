#!/bin/bash
# Stage 3, step 3: the Euler step sweep (10, 8, 6, 5), D43's protocol (five noise draws, Stage 1 and streaming, TT
# and the reference). 1. TT on the device, alone on the host (its timings are measurements). 2. The reference on
# the CPU, one job per step count in parallel. 3. Scoring. 10 steps: D43's draws (TT's are re-run here too, and
# must come out identical).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
P=/home/user/tt-metal/models/experimental/cosyvoice2/scripts
D=/home/user/data/cosyvoice2_steps
D43=/home/user/data/cosyvoice2_draws
export HF_HOME=/home/user/models
cd /home/user/tt-metal
if [ ! -f $RUN_DIR/tt_steps.exit ] || [ "$(cat $RUN_DIR/tt_steps.exit)" != 0 ]; then
  start_job tt_steps bash -c "echo \$\$ > $RUN_DIR/tt_steps.pid; exec /opt/venv/bin/python $S/steps_draws.py --out $D --steps 10,8,6,5"
  wait_job tt_steps 86400 || { grep -E 'Error|Traceback' -A3 $RUN_DIR/tt_steps.log | tail -12; exit 1; }
fi
grep -E "warm-ups|identical" $RUN_DIR/tt_steps.log
REF="env -u PYTHONPATH OMP_NUM_THREADS=24 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
cd /tmp
for k in 8 6 5; do
  cmd=""
  for s in 1 2 3 4 5; do
    cmd="$cmd $REF $S/ref_steps.py --steps $k $P/run_reference.py --out-dir $D/steps$k/ref_stage1_seed$s --noise-seed $s || exit 1;"
    cmd="$cmd $REF $S/ref_steps.py --steps $k $P/streaming_reference.py --inputs /home/user/data/cosyvoice2_inputs --tokens-from /home/user/data/cosyvoice2_runs/0929/stage1_head --out-dir $D/steps$k/ref_stream_seed$s --noise-seed $s || exit 1;"
  done
  start_job ref_steps$k bash -c "$cmd"
done
for k in 8 6 5; do wait_job ref_steps$k 86400 || { tail -20 $RUN_DIR/ref_steps$k.log; exit 1; }; done
g() { echo $(for s in 1 2 3 4 5; do echo "$1_seed$s"; done); }
groups=""
for mode in stage1 stream; do
  groups="$groups --group 'TT $mode 10' $(g $D43/tt_$mode) --group 'reference $mode 10' $(g $D43/ref_$mode)"
  for k in 8 6 5; do
    groups="$groups --group 'TT $mode $k' $(g $D/steps$k/tt_$mode) --group 'reference $mode $k' $(g $D/steps$k/ref_$mode)"
  done
done
start_job score_steps bash -c "exec env -u PYTHONPATH OMP_NUM_THREADS=48 LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models /home/user/cosyvoice2_ref_env/bin/python $P/eval_draws.py --reuse --out $RUN_DIR/steps.json $groups"
wait_job score_steps 86400
sed -n '/WER % and SIM/,$p' $RUN_DIR/score_steps.log
