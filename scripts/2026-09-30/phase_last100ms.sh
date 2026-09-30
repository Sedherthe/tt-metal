#!/bin/bash
# The stage A end gate's three checks (the last 20 ms level within 3 dB, the last 0.4 s difference 15 dB below the
# signal, the last 0.1 s difference 12 dB below it), on the fix and then on the old padding (old_padding_plugin.py:
# HiFTStream.step as it was before ed1c3ad1c5). The first run should pass and the second fail.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
T=models/experimental/cosyvoice2/tests/e2e/test_streaming.py
cd /home/user/tt-metal
start_job gate_last_fix bash -c "echo \$\$ > $RUN_DIR/gate_last_fix.pid; exec /opt/venv/bin/python -m pytest $T::test_stream_schedule_is_upstreams $T::test_device_offline_streaming_matches_upstream_streaming --timeout=0 -p no:cacheprovider -q -s"
wait_job gate_last_fix 86400
grep -E "\(final\)|passed|failed" $RUN_DIR/gate_last_fix.log | cut -c1-330
start_job gate_last_old bash -c "echo \$\$ > $RUN_DIR/gate_last_old.pid; PYTHONPATH=$S:\$PYTHONPATH exec /opt/venv/bin/python -m pytest -p old_padding_plugin $T::test_device_offline_streaming_matches_upstream_streaming --timeout=0 -p no:cacheprovider -q -s -vv"
wait_job gate_last_old 86400
grep -E "old_padding_plugin|\(final\)|passed|failed" $RUN_DIR/gate_last_old.log | cut -c1-330
grep -E "final chunk: last" $RUN_DIR/gate_last_old.log | sed 's/^E *//; s/^ *//' | sort -u | cut -c1-200
