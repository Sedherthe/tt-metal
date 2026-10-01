#!/bin/bash
# 2026-10-01 (B44): is the LLM bitwise reproducible on this card? The token-accuracy test (teacher-forced, no
# sampling) gave 96.34 % at 12:04 and 96.18 % at 13:37 (the record: 95.94 % on two pods), while the pipeline's
# sampled tokens equal the record again from 13:11. One process per step; nothing else on the host.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1001
D=/home/user/cosyvoice2-notes-wt/scripts/2026-10-01
T=models/experimental/cosyvoice2/tests
export HF_HOME=/home/user/models COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
export COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
PY=/opt/venv/bin/python
cd /home/user/tt-metal
start_job probe_b bash -c "echo \$\$ > $RUN_DIR/probe_b.pid; exec $PY $D/determinism_probe.py --repeats 200 --out $RUN_DIR/probe_b.json"
wait_job probe_b 86400; grep -E '^(matmul|softmax|rms|DETERM|NOT)' $RUN_DIR/probe_b.log
start_job llm_probe bash -c "echo \$\$ > $RUN_DIR/llm_probe.pid; exec $PY $D/llm_determinism_probe.py --repeats 8 --out $RUN_DIR/llm_probe.json"
wait_job llm_probe 86400; grep -E '^(traced|eager|LLM)|Error' $RUN_DIR/llm_probe.log | cut -c1-200
for k in 2 3; do
  start_job tokacc$k bash -c "echo \$\$ > $RUN_DIR/tokacc$k.pid; exec $PY -m pytest $T/e2e/test_token_accuracy.py --timeout=0 -p no:cacheprovider -q -s -rs"
  wait_job tokacc$k 86400; grep -E "margin at|token_accuracy >" $RUN_DIR/tokacc$k.log | tail -2
done
echo "llm probe done"
