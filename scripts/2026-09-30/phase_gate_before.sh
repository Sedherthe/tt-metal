#!/bin/bash
# D41's gates on the current HiFT (end padding with silence, no masks): both should fail. The Stage 1 end gate
# (tests/pcc/test_hift_masked.py, with the host proof of the masking rules) and the stage A streaming gate.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
T=models/experimental/cosyvoice2/tests
cd /home/user/tt-metal
n0=$(find $KC -name "*.elf" | wc -l)
start_job gate_before bash -c "echo \$\$ > $RUN_DIR/gate_before.pid; exec /opt/venv/bin/python -m pytest $T/pcc/test_hift_masked.py $T/e2e/test_streaming.py::test_device_offline_streaming_matches_upstream_streaming --timeout=0 -p no:cacheprovider -q -s"
wait_job gate_before 86400
echo "binaries compiled by gate_before: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
grep -E "^\| zero|^E .*(last 20 ms|PCC)|passed|failed" $RUN_DIR/gate_before.log | cut -c1-260 | tail -40
