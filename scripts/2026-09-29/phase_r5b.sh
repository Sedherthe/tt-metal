#!/bin/bash
# R5b: the streaming guard. The interleaved test with its new first check (before warmup_streaming(), the call is
# refused before any device work), untracked, alone; with the host-only schedule test.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
T=models/experimental/cosyvoice2/tests/e2e/test_streaming.py
cd /home/user/tt-metal
n0=$(find $KC -name "*.elf" | wc -l)
start_job r5_guard bash -c "echo \$\$ > $RUN_DIR/r5_guard.pid; exec /opt/venv/bin/python -m pytest $T::test_stream_schedule_is_upstreams $T::test_device_streaming_interleaved_with_llm --timeout=0 -p no:cacheprovider -q -s"
wait_job r5_guard 86400
rc=$?
echo "binaries compiled by r5_guard: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
grep -E "EXPECTED_ERROR|chunks,|chunk offset|passed|failed|^E " $RUN_DIR/r5_guard.log | cut -c1-220 | tail -14
exit $rc
