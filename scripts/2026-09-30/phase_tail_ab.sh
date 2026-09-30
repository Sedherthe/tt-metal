#!/bin/bash
# The "you" clip, device side: you_tail_ab.py (the final HiFT call end-padded vs exact length), alone on the card,
# with the kernel binaries it compiles counted. Then both corpus runs scored as usual (reference venv, CPU).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
OUT=$RUN_DIR/tail_ab
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
cd /home/user/tt-metal
n0=$(find $KC -name "*.elf" | wc -l)
start_job tail_ab bash -c "echo \$\$ > $RUN_DIR/tail_ab.pid; exec /opt/venv/bin/python /home/user/cosyvoice2-notes-wt/scripts/2026-09-30/you_tail_ab.py --out $OUT"
wait_job tail_ab 86400 || { echo "tail_ab failed"; grep -E '^E |Error|Traceback' -A3 $RUN_DIR/tail_ab.log | tail -20; exit 1; }
echo "binaries compiled by tail_ab: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
grep -E '^(padded|exact) |sanity|equals' $RUN_DIR/tail_ab.log | cut -c1-400
for v in padded exact; do
  start_job score_$v bash -c "cd /tmp && env -u PYTHONPATH OMP_NUM_THREADS=48 /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $OUT/${v}_1987 --baseline /home/user/data/cosyvoice2_streaming_ref"
  wait_job score_$v 7200
  grep -vE "Loading|it/s|iB/s|Warning|warn|mask" $RUN_DIR/score_$v.log | tail -12
done
