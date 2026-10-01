#!/bin/bash
# 2026-10-01: launches the reference chain and the device chain as sentinel jobs (phase_ref.exit, phase_device.exit).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1001
D=/home/user/cosyvoice2-notes-wt/scripts/2026-10-01
start_job phase_ref $D/phase_ref.sh
start_job phase_device $D/phase_device.sh
wait_job phase_ref 172800
wait_job phase_device 172800
