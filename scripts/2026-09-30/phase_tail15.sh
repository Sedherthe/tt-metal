#!/bin/bash
# Stage A at the last 0.4 s's new threshold (15 dB below the signal; tail_margin.py's 36-chunk distribution).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
T=models/experimental/cosyvoice2/tests/e2e/test_streaming.py
cd /home/user/tt-metal
start_job gate_tail15 bash -c "echo \$\$ > $RUN_DIR/gate_tail15.pid; exec /opt/venv/bin/python -m pytest $T::test_stream_schedule_is_upstreams $T::test_device_offline_streaming_matches_upstream_streaming --timeout=0 -p no:cacheprovider -q -s"
wait_job gate_tail15 86400
grep -E "\(final\)|passed|failed|^E " $RUN_DIR/gate_tail15.log | cut -c1-260 | tail -12
