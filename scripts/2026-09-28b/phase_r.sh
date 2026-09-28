#!/bin/bash
# Stage 1 re-verified on chunked HiFT: cold start (empty kernel cache) and an identical second process, then the
# Stage 1 demo and the perf test on that cache, then WER/SIM on the demo's audio (CPU, nothing timed alongside).
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928b
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
export TT_METAL_CACHE=$RUN_DIR/kcache HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
cd /home/user/tt-metal
[ -e $TT_METAL_CACHE ] && { echo "$TT_METAL_CACHE exists; the cold run needs an empty cache"; exit 2; }
mkdir -p $TT_METAL_CACHE
elfs() { find $TT_METAL_CACHE -name "*.elf" | wc -l; }
for label in chunked-cold chunked-second; do
  start_job warm_$label bash -c "source python_env/bin/activate && python $SCR/warmup_measure.py --label $label --out $RUN_DIR/warm_$label.json"
  wait_job warm_$label 10800 || { echo "stopping at warm_$label"; exit 1; }
  grep -E '^\{"label' $RUN_DIR/warm_$label.log | cut -c1-1600
done
n0=$(elfs)
start_job stage1_chunked bash -c "source python_env/bin/activate && python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS --out $RUN_DIR/stage1_chunked --warmup buckets"
wait_job stage1_chunked 7200 || exit 1
n1=$(elfs); echo "binaries compiled by the Stage 1 demo: $((n1 - n0))"
tail -12 $RUN_DIR/stage1_chunked.log | grep -E "^\||Distinct"
start_job stage1_perf bash -c "source python_env/bin/activate && python -m pytest models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job stage1_perf 7200
n2=$(elfs); echo "binaries compiled by the perf test: $((n2 - n1))"
grep -E "start-up|^  \||PASS|MISS|passed|failed" $RUN_DIR/stage1_perf.log | tail -12
start_job score_chunked bash -c "cd /tmp && /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $RUN_DIR/stage1_chunked --baseline /home/user/data/cosyvoice2_runs/reference"
wait_job score_chunked 7200
grep -vE "Loading|it/s|iB/s|Warning|mask" $RUN_DIR/score_chunked.log | tail -14
