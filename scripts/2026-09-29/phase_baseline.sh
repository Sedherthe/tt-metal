#!/bin/bash
# 2026-09-29: Stage 1 baseline on HEAD (7bd094cc3e), before R1. The demo's wavs and tokens are the bit-exact baseline
# R1 is compared against; its timing re-checks Stage 1 on this pod. Scoring (CPU) runs only after the demo has exited.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
OUT=$RUN_DIR/stage1_head
cd /home/user/tt-metal
[ -e $OUT ] && { echo "$OUT exists"; exit 2; }
elfs() { find $KC -name "*.elf" | wc -l; }
n0=$(elfs); echo "$(date '+%F %T') HEAD $(git rev-parse --short HEAD); binaries on disk: $n0"
start_job stage1_head bash -c "echo \$\$ > $RUN_DIR/stage1_head.pid; exec /opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs /home/user/data/cosyvoice2_inputs --out $OUT --warmup buckets"
wait_job stage1_head 86400 || exit 1
n1=$(elfs); echo "$(date '+%F %T') binaries compiled by the demo: $((n1 - n0))"
grep -E "^\||Distinct|start-up|warm" $RUN_DIR/stage1_head.log | tail -14
start_job score_head bash -c "cd /tmp && env -u PYTHONPATH OMP_NUM_THREADS=48 HF_HOME=/home/user/models /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $OUT --baseline /home/user/data/cosyvoice2_runs/reference"
wait_job score_head 7200
grep -vE "Loading|it/s|iB/s|Warning|warn|mask" $RUN_DIR/score_head.log | tail -16
