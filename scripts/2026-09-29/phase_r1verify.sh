#!/bin/bash
# R1 verification: the Stage 1 demo with the ROW_MAJOR candidate (alone: it is timed), bit-compared with the baseline
# on HEAD; then the device suite, while R2's reference half runs on CPU (the suite times nothing).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
export COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref
OUT=$RUN_DIR/stage1_r1
T=models/experimental/cosyvoice2/tests
PERF="$T/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances[device_params0]"
cd /home/user/tt-metal
[ -e $OUT ] && { echo "$OUT exists"; exit 2; }
elfs() { find $KC -name "*.elf" | wc -l; }
n0=$(elfs); echo "$(date '+%F %T') tree: $(git rev-parse --short HEAD) + uncommitted R1; binaries on disk: $n0"
start_job stage1_r1 bash -c "echo \$\$ > $RUN_DIR/stage1_r1.pid; exec /opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS --out $OUT --warmup buckets"
wait_job stage1_r1 86400 || exit 1
n1=$(elfs); echo "$(date '+%F %T') binaries compiled by the demo: $((n1 - n0))"
grep -E "Distinct" $RUN_DIR/stage1_r1.log
grep -E "fast conv path disagrees" $RUN_DIR/stage1_r1.log | sed -E 's/.*_verify_and_resolve:[0-9]+ - //'
python3 - <<'EOF'
import json, numpy as np, soundfile
base, new = "/home/user/data/cosyvoice2_runs/0929/stage1_head", "/home/user/data/cosyvoice2_runs/0929/stage1_r1"
rb = {r["case_id"]: r for r in json.load(open(f"{base}/results.json"))["results"]}
rn = {r["case_id"]: r for r in json.load(open(f"{new}/results.json"))["results"]}
assert rb.keys() == rn.keys()
for cid in rb:
    same_tokens = rb[cid]["segment_tokens"] == rn[cid]["segment_tokens"]
    a, _ = soundfile.read(f"{base}/{rb[cid]['wav']}", dtype="float32")
    b, _ = soundfile.read(f"{new}/{rn[cid]['wav']}", dtype="float32")
    same_len = a.shape == b.shape
    diff = float(np.abs(a - b).max()) if same_len else float("nan")
    print(f"  {cid}: tokens identical {same_tokens}, audio {'bit-identical' if same_len and diff == 0 else f'max|diff| {diff}'}")
EOF
start_job r2_ref bash -c "cd /tmp && env -u PYTHONPATH OMP_NUM_THREADS=32 /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/hift_streaming_reference.py --out-dir /home/user/data/cosyvoice2_hift_stream_ref_r2"
n2=$(elfs)
start_job suite_r1 bash -c "echo \$\$ > $RUN_DIR/suite_r1.pid; exec /opt/venv/bin/python -m pytest $T --timeout=0 -p no:cacheprovider -q -rs --durations=10 --deselect '$PERF'"
wait_job r2_ref 7200
grep -E "^  [0-9]|Error|Traceback" $RUN_DIR/r2_ref.log | tail -8
wait_job suite_r1 86400
n3=$(elfs); echo "$(date '+%F %T') binaries compiled by the suite: $((n3 - n2))"
grep -E "passed|failed|^FAILED|^ERROR" $RUN_DIR/suite_r1.log | tail -6
