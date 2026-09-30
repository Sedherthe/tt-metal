#!/bin/bash
# B28's fix candidate on the card: you_tail_front.py (the final HiFT call padded in front), alone, kernels counted;
# then its corpus run scored as usual (reference venv, CPU).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
OUT=$RUN_DIR/tail_ab
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
cd /home/user/tt-metal
n0=$(find $KC -name "*.elf" | wc -l)
start_job tail_front bash -c "echo \$\$ > $RUN_DIR/tail_front.pid; exec /opt/venv/bin/python /home/user/cosyvoice2-notes-wt/scripts/2026-09-30/you_tail_front.py --out $OUT"
wait_job tail_front 86400 || { echo "tail_front failed"; grep -E '^E |Error|Traceback' -A3 $RUN_DIR/tail_front.log | tail -20; exit 1; }
echo "binaries compiled by tail_front: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
grep -E '^front ' $RUN_DIR/tail_front.log | cut -c1-400
start_job score_front bash -c "cd /tmp && env -u PYTHONPATH OMP_NUM_THREADS=48 /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $OUT/front_1987 --baseline /home/user/data/cosyvoice2_streaming_ref"
wait_job score_front 7200
grep -vE "Loading|it/s|iB/s|Warning|warn|mask" $RUN_DIR/score_front.log | tail -12
