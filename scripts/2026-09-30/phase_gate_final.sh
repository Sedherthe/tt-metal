#!/bin/bash
# D41's final stage A criteria (the last 20 ms level, the last 0.4 s difference, the final chunk's PCC before those
# 0.4 s) and the "you" clip's device half at 3 dB, on the masked HiFT.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
T=models/experimental/cosyvoice2/tests/e2e/test_streaming.py
cd /home/user/tt-metal
start_job gate_final bash -c "echo \$\$ > $RUN_DIR/gate_final.pid; exec /opt/venv/bin/python -m pytest $T::test_device_offline_streaming_matches_upstream_streaming $T::test_device_you_clip_noise_draws --timeout=0 -p no:cacheprovider -q -s"
wait_job gate_final 86400
grep -E "\(final\)|dBFS, upstream's|passed|failed|^E " $RUN_DIR/gate_final.log | cut -c1-260 | tail -24
