#!/bin/bash
# The "you" clip, scorer side (CPU, reference venv): five runs as the scorer normally runs (48 threads, as the scoring
# job ran), the first with the tail splices; five seeded; then 24 and 1 threads, to see whether CPU numerics move it.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
OUT=$RUN_DIR/you_asr.jsonl
PY="env -u PYTHONPATH /home/user/cosyvoice2_ref_env/bin/python /home/user/cosyvoice2-notes-wt/scripts/2026-09-30/you_asr.py"
cd /tmp/claude-1000/-home-user-tt-metal/5b576035-d516-4f80-869b-0f1cc86dd847/scratchpad
rm -f $OUT
for i in 1 2 3 4 5; do
  extra=""; [ $i = 1 ] && extra="--splices"
  start_job you_normal$i bash -c "OMP_NUM_THREADS=48 exec $PY --mode normal --tag normal$i $extra --out $OUT"
  wait_job you_normal$i 7200
done
for i in 1 2 3 4 5; do
  start_job you_seeded$i bash -c "OMP_NUM_THREADS=48 exec $PY --mode seeded --tag seeded$i --out $OUT"
  wait_job you_seeded$i 7200
done
for t in 24 1; do
  start_job you_threads$t bash -c "OMP_NUM_THREADS=$t exec $PY --mode normal --tag threads$t --out $OUT"
  wait_job you_threads$t 7200
done
grep -h -E "^\[" $RUN_DIR/you_*.log
