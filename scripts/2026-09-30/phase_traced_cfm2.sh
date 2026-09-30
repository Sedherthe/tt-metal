#!/bin/bash
# phase_traced_cfm.sh's proof and control again, after traced_cfm_prototype.py learned that a bucket where some conv's
# check chose the raw (host) weight cannot be traced: the first proof run's capture at 5,120 frames failed with
# "Writes are not supported during trace capture". Such buckets now stay eager.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
export HF_HOME=/home/user/models
cd /home/user/tt-metal
start_job traced_cfm_proof2 bash -c "echo \$\$ > $RUN_DIR/traced_cfm_proof2.pid; TT_METAL_TRACE_ALLOC_TRACKING=1 exec /opt/venv/bin/python $S/traced_cfm_prototype.py --order cfm-first --llm-output preallocated --out $RUN_DIR/traced_cfm_proof2.json"
wait_job traced_cfm_proof2 86400
grep -E "untraceable|trace region|\[traced\]|\"case\"|Traceback" $RUN_DIR/traced_cfm_proof2.log | cut -c1-300 | tail -20
start_job traced_cfm_control2 bash -c "echo \$\$ > $RUN_DIR/traced_cfm_control2.pid; TT_METAL_TRACE_ALLOC_TRACKING=1 exec /opt/venv/bin/python $S/traced_cfm_prototype.py --order cfm-first --llm-output inside --out $RUN_DIR/traced_cfm_control2.json"
wait_job traced_cfm_control2 86400
grep -E "untraceable|trace region|\[traced\]|Traceback" $RUN_DIR/traced_cfm_control2.log | cut -c1-400 | tail -10
