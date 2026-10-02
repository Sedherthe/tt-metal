#!/bin/bash
# 2026-10-02: the issue draft's repro.py exactly as it will be posted (extracted from the draft), five fresh processes, an empty TT_METAL_CACHE.
# on this pod, the five invocations in the draft's order, each a fresh process, against a new empty TT_METAL_CACHE.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1002
KC=/home/user/data/kc_repro_1002_issue
R=/home/user/data/cosyvoice2_runs/1002/issue_repro.py
[ -e $KC ] && { echo "$KC exists; refusing to reuse a kernel cache"; exit 2; }
mkdir -p $KC; cd /tmp
echo "tt-metal $(git -C /home/user/tt-metal rev-parse --short HEAD) (C++ base $(git -C /home/user/tt-metal merge-base HEAD origin/main | cut -c1-10)), KMD $(cat /sys/module/tenstorrent/version)"
i=0
for args in "dram 0" "dram 0" "dram 1" "l1 0" "l1 1"; do
  i=$((i + 1))
  start_job issue_repro$i bash -c "echo \$\$ > $RUN_DIR/issue_repro$i.pid; exec env TT_METAL_CACHE=$KC /opt/venv/bin/python $R $args"
  wait_job issue_repro$i 3600 || exit 1
  grep -E "^config tensors" $RUN_DIR/issue_repro$i.log
done
echo "binaries in $KC: $(find $KC -name '*.elf' | wc -l)"
