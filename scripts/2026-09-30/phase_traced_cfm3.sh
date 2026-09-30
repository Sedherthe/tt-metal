#!/bin/bash
# The traced-CFM proposal's evidence, final (traced_cfm_prototype.py after the fixes that phase_traced_cfm.sh and
# phase_traced_cfm2.sh, and the diagnostics between them, called for):
# - buckets with a raw-weight conv verdict (host weight: conv1d writes on every call) stay eager;
# - `inference_streaming`'s device locals are freed before the CFM runs (the three buffers the tracker named);
# - audio is compared by log-mel L1, not waveform PCC.
#   A. the proof: the allocation tracker on, all six utterances;
#   B. the negative control: the decode trace's logits allocated inside its capture; the tracker must flag them;
#   C. the timing: the tracker off, all six, against the eager pipeline in the same process, wavs written.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
P=$S/traced_cfm_prototype.py
export HF_HOME=/home/user/models
cd /home/user/tt-metal
start_job tcfm_proof bash -c "echo \$\$ > $RUN_DIR/tcfm_proof.pid; TT_METAL_TRACE_ALLOC_TRACKING=1 exec /opt/venv/bin/python $P --order cfm-first --llm-output preallocated --out $RUN_DIR/tcfm_proof.json"
wait_job tcfm_proof 86400
grep -E "untraceable|trace region|\[traced\]|^\{\"case\"|Traceback" $RUN_DIR/tcfm_proof.log | cut -c1-300 | tail -16
start_job tcfm_control bash -c "echo \$\$ > $RUN_DIR/tcfm_control.pid; TT_METAL_TRACE_ALLOC_TRACKING=1 exec /opt/venv/bin/python $P --order cfm-first --llm-output inside --no-baseline --limit 1 --out $RUN_DIR/tcfm_control.json"
wait_job tcfm_control 86400
grep -E "trace region|\[traced\]|Traceback" $RUN_DIR/tcfm_control.log | cut -c1-400 | tail -6
start_job tcfm_timing bash -c "echo \$\$ > $RUN_DIR/tcfm_timing.pid; exec /opt/venv/bin/python $P --order cfm-first --llm-output preallocated --wav-dir $RUN_DIR/tcfm_wavs --out $RUN_DIR/tcfm_timing.json"
wait_job tcfm_timing 86400
grep -E "trace region|\[eager\]|\[traced\]|^\{\"case\"" $RUN_DIR/tcfm_timing.log | cut -c1-330 | tail -16
