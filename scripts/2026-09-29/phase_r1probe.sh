#!/bin/bash
# R1 re-measurement: #36487's reproducer three ways, then the rebuild spec's geometry table three ways (standalone).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
N=/home/user/cosyvoice2-notes-wt/scripts/2026-09-29
cd $N
start_job r1_repro bash -c "echo \$\$ > $RUN_DIR/r1_repro.pid; exec env HF_HOME=/home/user/models /opt/venv/bin/python r1_repro_36487.py"
wait_job r1_repro 86400
grep -E "PCC" $RUN_DIR/r1_repro.log
start_job r1_layout bash -c "echo \$\$ > $RUN_DIR/r1_layout.pid; exec /opt/venv/bin/python r1_prepare_layout.py"
wait_job r1_layout 86400
sed -n '/^| case/,$p' $RUN_DIR/r1_layout.log
