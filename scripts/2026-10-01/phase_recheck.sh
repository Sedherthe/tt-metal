#!/bin/bash
# 2026-10-01 (B44): is the card reproducible now? 11:05-12:04 gave different tokens and logits from the record;
# the perf tests at 13:11 and 13:15 gave the recorded tokens. Each step is its own process; nothing else on the host.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1001
D=/home/user/cosyvoice2-notes-wt/scripts/2026-10-01
T=models/experimental/cosyvoice2/tests
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
PY=/opt/venv/bin/python
cd /home/user/tt-metal
tokens() { python3 -c "
import json; r = json.load(open('$1/results.json'))['results']
print('tokens per case:', '/'.join(str(sum(len(s) for s in c['segment_tokens'])) for c in r), '(record: 213/95/347/75/317/202)')"; }
start_job probe bash -c "echo \$\$ > $RUN_DIR/probe.pid; exec $PY $D/determinism_probe.py --repeats 200 --out $RUN_DIR/probe.json"
wait_job probe 86400; grep -vE "DEBUG|info |warning|^Config" $RUN_DIR/probe.log | tail -6
start_job tt_smi4 /home/user/.local/bin/tt-smi -s; wait_job tt_smi4 1800
start_job tokacc bash -c "echo \$\$ > $RUN_DIR/tokacc.pid; exec $PY -m pytest $T/e2e/test_token_accuracy.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job tokacc 86400; grep -E "set |positions, top-1|margin at|token_accuracy >|passed|failed" $RUN_DIR/tokacc.log | tail -8
[ -e $RUN_DIR/stage1_b ] && { echo "stage1_b exists"; exit 2; }
start_job stage1_b bash -c "echo \$\$ > $RUN_DIR/stage1_b.pid; exec $PY models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS --out $RUN_DIR/stage1_b --warmup buckets"
wait_job stage1_b 86400; grep -E "^Distinct" $RUN_DIR/stage1_b.log | tail -1; tokens $RUN_DIR/stage1_b
start_job api bash -c "echo \$\$ > $RUN_DIR/api.pid; exec $PY -m pytest $T/e2e/test_pipeline_api.py --timeout=0 -p no:cacheprovider -q -s -rs"
wait_job api 86400; grep -E "^\| [0-9] |repeat of|tokens differ|passed|failed" $RUN_DIR/api.log | cut -c1-200 | tail -14
start_job tt_smi5 /home/user/.local/bin/tt-smi -s; wait_job tt_smi5 1800
echo "recheck done"
