#!/bin/bash
# 2026-10-01: re-verify the backup tip (213afe7909) on this pod's card (0000:e1:00.0), on an empty kernel cache.
# 1. the Stage 1 demo, once the inputs exist (it fills the kernel cache; the reference chain shares the CPU, so its
#    timings are not measurements; its tokens are checked against 09-29's lengths);
# 2. after the reference chain: upstream's streaming of the demo's tokens (reference venv), the suite's stage A reference;
# 3. the device suite with every reference (the perf file deselected), the demo's WER/SIM scored on the CPU alongside;
# 4. the Stage 1 perf test, then the streaming perf test, each in its own process, nothing else on the host.
# Restarted 10:58 after the harness killed the first run (B43); start it with detach.sh.
# No timeouts: wait_job only reports. Each device job's pid goes to <name>.pid, for SIGINT only.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1001
KC=/home/user/.cache/tt-metal-cache
S=/home/user/tt-metal/models/experimental/cosyvoice2/scripts
T=models/experimental/cosyvoice2/tests
PERF=$T/perf/test_pipeline_perf.py
REFPY="env -u PYTHONPATH OMP_NUM_THREADS=48 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
OUT=$RUN_DIR/stage1
SREF=/home/user/data/cosyvoice2_streaming_ref
for d in $OUT $SREF; do [ -e $d ] && { echo "$d exists; refusing to mix runs"; exit 2; }; done
elfs() { find $KC -name "*.elf" | wc -l; }
cd /home/user/tt-metal
echo "$(date '+%F %T') HEAD $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l) modified files"

wait_job inputs_tfx 86400 || { echo "inputs failed; not starting"; exit 1; }
n0=$(elfs); echo "binaries on disk: $n0"
start_job stage1 bash -c "echo \$\$ > $RUN_DIR/stage1.pid; exec /opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS --out $OUT --warmup buckets"
wait_job stage1 86400 || exit 1
n1=$(elfs); echo "binaries compiled by the demo: $((n1 - n0))"
grep -E "^\||Distinct|start-up|warm" $RUN_DIR/stage1.log | tail -14
python3 -c "
import json; r = json.load(open('$OUT/results.json'))
cases = r['cases'] if isinstance(r, dict) and 'cases' in r else r
print('tokens per case:', [sum(len(s) for s in c['segment_tokens']) for c in cases], '(09-29: 213/95/347/75/317/202)')"

wait_job phase_ref 86400
for step in inputs inputs_tfx reference token_ref token_ref_tfx hift_stream_ref; do
  [ "$(cat $RUN_DIR/$step.exit 2>/dev/null)" = 0 ] || { echo "reference step $step did not succeed; not starting the suite"; exit 1; }
done
start_job stream_ref bash -c "cd /tmp && exec $REFPY $S/streaming_reference.py --inputs $COSYVOICE2_INPUTS --tokens-from $OUT --out-dir $SREF"
wait_job stream_ref 14400 || exit 1

export COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
export COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref COSYVOICE2_STREAM_REF=$SREF
(cd /tmp && start_job score_stage1 bash -c "exec $REFPY $S/eval_wer_sim.py --run-dir $OUT --baseline /home/user/data/cosyvoice2_runs/reference")
n0=$(elfs)
start_job suite bash -c "echo \$\$ > $RUN_DIR/suite.pid; exec /opt/venv/bin/python -m pytest $T --timeout=0 -p no:cacheprovider -q -rs --durations=15 --deselect $PERF"
wait_job suite 86400
n1=$(elfs); echo "binaries compiled by the suite: $((n1 - n0))"
grep -E "passed|failed|^FAILED|^ERROR|^SKIPPED" $RUN_DIR/suite.log | cut -c1-220 | tail -15
wait_job score_stage1 86400
grep -vE "Loading|it/s|iB/s|Warning|warn|mask" $RUN_DIR/score_stage1.log | tail -16

for p in nonstreaming_rtf_distinct_utterances streaming_first_audio_and_rtf_distinct_utterances; do
  n0=$(elfs)
  start_job perf_$p bash -c "echo \$\$ > $RUN_DIR/perf_$p.pid; exec /opt/venv/bin/python -m pytest $PERF::test_device_$p --timeout=0 -p no:cacheprovider -q -s -rs"
  wait_job perf_$p 86400
  echo "binaries compiled by perf_$p: $(( $(elfs) - n0 ))"
  grep -E "start-up|^  \||ttfp_ms|rtf_|PASS|MISS|Meets|Misses|passed|failed|^E " $RUN_DIR/perf_$p.log | cut -c1-220 | tail -20
done
echo "device chain done"
