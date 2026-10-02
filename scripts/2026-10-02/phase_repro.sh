#!/bin/bash
# 2026-10-02: the kernel-cache issue draft's minimal reproducer (drafts/2026-09-27_ttnn_issue_conv_dram_config_kernel_hash.md)
# on this pod, the five invocations in the draft's order, each a fresh process, against a new empty TT_METAL_CACHE.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1002
KC=/home/user/data/kc_repro_1002
R=/home/user/cosyvoice2-notes-wt/scripts/2026-10-02/repro_conv_dram_config_kernel_hash.py
[ -e $KC ] && { echo "$KC exists; refusing to reuse a kernel cache"; exit 2; }
mkdir -p $KC; cd /tmp
echo "tt-metal $(git -C /home/user/tt-metal rev-parse --short HEAD) (C++ base $(git -C /home/user/tt-metal merge-base HEAD origin/main | cut -c1-10)), KMD $(cat /sys/module/tenstorrent/version)"
i=0
for args in "dram 0" "dram 0" "dram 1" "l1 0" "l1 1"; do
  i=$((i + 1))
  start_job repro$i bash -c "echo \$\$ > $RUN_DIR/repro$i.pid; exec env TT_METAL_CACHE=$KC /opt/venv/bin/python $R $args"
  wait_job repro$i 3600 || exit 1
  grep -E "^config tensors" $RUN_DIR/repro$i.log
done
echo "binaries in $KC: $(find $KC -name '*.elf' | wc -l)"
