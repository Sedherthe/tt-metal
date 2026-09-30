#!/bin/bash
# R6 on the masked HiFT. The streaming perf test (Stage 3's figures enforced, tests/perf/gates.py); start-up cold and
# warm (startup_measure.py twice against one new TT_METAL_CACHE: both warm-ups); the non-streaming cold first request
# (demo.py --warmup none, one utterance, fresh processes) on an empty kernel cache and on the cache the warmed runs
# filled (B22: the spec's RTF 32.5 and 64.3 were unverified). Nothing else runs on the host meanwhile.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
KC_NEW=/home/user/data/kc_r6_startup
KC_FIRST=/home/user/data/kc_r6_first
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
DEMO="/opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS"
elfs() { find "$1" -name "*.elf" 2>/dev/null | wc -l; }
for d in $KC_NEW $KC_FIRST; do [ -e $d ] && { echo "$d exists; refusing to reuse a kernel cache"; exit 2; }; done
cd /home/user/tt-metal
n0=$(elfs $KC)
start_job perf_stream bash -c "echo \$\$ > $RUN_DIR/perf_stream.pid; exec /opt/venv/bin/python -m pytest models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_streaming_first_audio_and_rtf_distinct_utterances --timeout=0 -p no:cacheprovider -q -s"
wait_job perf_stream 86400
echo "binaries compiled by perf_stream: $(( $(elfs $KC) - n0 ))"
grep -E "start-up|^  \| zero|ttfp_ms|rtf_streaming|lever|passed|failed|^E " $RUN_DIR/perf_stream.log | cut -c1-220 | tail -16
mkdir -p $KC_NEW
for label in cold warm; do
  start_job startup_$label bash -c "echo \$\$ > $RUN_DIR/startup_$label.pid; exec env TT_METAL_CACHE=$KC_NEW /opt/venv/bin/python $S/startup_measure.py --label $label --out $RUN_DIR/startup_$label.json"
  wait_job startup_$label 86400
  tail -1 $RUN_DIR/startup_$label.log | cut -c1-600
done
mkdir -p $KC_FIRST
start_job first_empty bash -c "echo \$\$ > $RUN_DIR/first_empty.pid; exec env TT_METAL_CACHE=$KC_FIRST $DEMO --warmup none --cases zero_shot_121-127105-0003 --out $RUN_DIR/first_request_empty_cache"
wait_job first_empty 86400
echo "binaries compiled by first_empty: $(elfs $KC_FIRST)"
grep -E "^\| zero|Distinct" $RUN_DIR/first_empty.log | tail -2
n0=$(elfs $KC)
start_job first_filled bash -c "echo \$\$ > $RUN_DIR/first_filled.pid; exec $DEMO --warmup none --cases zero_shot_121-127105-0003 --out $RUN_DIR/first_request_filled_cache"
wait_job first_filled 86400
echo "binaries compiled by first_filled: $(( $(elfs $KC) - n0 ))"
grep -E "^\| zero|Distinct" $RUN_DIR/first_filled.log | tail -2
