#!/bin/bash
# R5: streaming measured. It starts only once R4 is committed (r4_committed.exit, written by hand after the push), so
# no pre-commit stash can race a demo's imports. Two warm runs (fresh processes, the streaming demo on the six
# distinct utterances), then the cold first request (no warm-up, one utterance) under the allocation tracker: its
# kernels compile and its convs are verified while the decode trace is alive. Then WER/SIM on CPU.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
IN=/home/user/data/cosyvoice2_inputs
DEMO="/opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs $IN --stream"
cd /home/user/tt-metal
wait_job r4_committed 86400
elfs() { find $KC -name "*.elf" | wc -l; }
for run in warm1 warm2; do
  n0=$(elfs)
  start_job r5_$run bash -c "echo \$\$ > $RUN_DIR/r5_$run.pid; exec $DEMO --out $RUN_DIR/r5_stream_$run"
  wait_job r5_$run 86400 || { echo "r5_$run failed"; exit 1; }
  echo "binaries compiled by r5_$run: $(( $(elfs) - n0 ))"
  grep -E "^\| zero|Distinct|warmed" $RUN_DIR/r5_$run.log | tail -10
done
n0=$(elfs)
start_job r5_cold bash -c "echo \$\$ > $RUN_DIR/r5_cold.pid; exec env TT_METAL_TRACE_ALLOC_TRACKING=1 $DEMO --warmup none --cases zero_shot_121-127105-0003 --out $RUN_DIR/r5_stream_cold"
wait_job r5_cold 86400 || { echo "r5_cold failed"; grep -E '^E |FATAL|Error' $RUN_DIR/r5_cold.log | head -5; }
echo "binaries compiled by r5_cold: $(( $(elfs) - n0 ))"
grep -E "^\| zero|Distinct" $RUN_DIR/r5_cold.log | tail -3
start_job r5_score bash -c "cd /tmp && env -u PYTHONPATH OMP_NUM_THREADS=48 /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $RUN_DIR/r5_stream_warm1 --baseline /home/user/data/cosyvoice2_streaming_ref"
wait_job r5_score 7200
grep -vE "Loading|it/s|iB/s|Warning|warn|mask" $RUN_DIR/r5_score.log | tail -12
