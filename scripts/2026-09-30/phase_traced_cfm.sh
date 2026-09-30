#!/bin/bash
# Stage 3 lever (b), the proposal's evidence (traced_cfm_prototype.py; the PR is unchanged):
#   A. the allocation tracker on: every streaming bucket's CFM trace captured at start-up, then the LLM decode trace,
#      every persistent buffer allocated first. It must stream the six with no tracker failure.
#   B. the negative control: the same, but the decode trace's logits allocated inside its capture, as today. The
#      tracker must flag it.
#   C. the tracker off: A's design timed (first audio, RTF), against the eager pipeline in the same process.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
export HF_HOME=/home/user/models
cd /home/user/tt-metal
start_job traced_cfm_proof bash -c "echo \$\$ > $RUN_DIR/traced_cfm_proof.pid; TT_METAL_TRACE_ALLOC_TRACKING=1 exec /opt/venv/bin/python $S/traced_cfm_prototype.py --order cfm-first --llm-output preallocated --out $RUN_DIR/traced_cfm_proof.json"
wait_job traced_cfm_proof 86400
grep -E "trace region|\[traced\]|\"case\"|Traceback|Error" $RUN_DIR/traced_cfm_proof.log | cut -c1-300 | tail -20
start_job traced_cfm_control bash -c "echo \$\$ > $RUN_DIR/traced_cfm_control.pid; TT_METAL_TRACE_ALLOC_TRACKING=1 exec /opt/venv/bin/python $S/traced_cfm_prototype.py --order cfm-first --llm-output inside --out $RUN_DIR/traced_cfm_control.json"
wait_job traced_cfm_control 86400
grep -E "trace region|\[traced\]|Traceback" $RUN_DIR/traced_cfm_control.log | cut -c1-400 | tail -10
start_job traced_cfm_timing bash -c "echo \$\$ > $RUN_DIR/traced_cfm_timing.pid; exec /opt/venv/bin/python $S/traced_cfm_prototype.py --order cfm-first --llm-output preallocated --out $RUN_DIR/traced_cfm_timing.json"
wait_job traced_cfm_timing 86400
grep -E "trace region|\[eager\]|\[traced\]|\"case\"" $RUN_DIR/traced_cfm_timing.log | cut -c1-300 | tail -20
