#!/bin/bash
# 2026-10-01, RUNBOOK section 1 on the new pod: tt-smi -s, the smoke test, then tt-smi -s again (heartbeat must advance).
# No timeouts; wait_job only reports.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1001
cd /tmp
start_job tt_smi1 /home/user/.local/bin/tt-smi -s
wait_job tt_smi1 1800 || exit 1
start_job smoke bash -c "echo \$\$ > $RUN_DIR/smoke.pid; exec /opt/venv/bin/python /home/user/cosyvoice2-notes-wt/scripts/2026-09-29/smoke_add.py"
wait_job smoke 1800 || exit 1
sleep 120
start_job tt_smi2 /home/user/.local/bin/tt-smi -s
wait_job tt_smi2 1800 || exit 1
echo "card check done"
