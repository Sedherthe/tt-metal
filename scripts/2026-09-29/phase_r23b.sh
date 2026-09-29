#!/bin/bash
# R2 and R3 gates re-run with their thresholds set from the first measurement, then a regression pass over the
# encoder and flow tests (R3 changed the encoder's streaming path).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
T=models/experimental/cosyvoice2/tests
cd /home/user/tt-metal
start_job r2_seams2 bash -c "echo \$\$ > $RUN_DIR/r2_seams2.pid; exec env COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref_r2 /opt/venv/bin/python -m pytest $T/pcc/test_hift_chunked.py --timeout=0 -p no:cacheprovider -q"
wait_job r2_seams2 86400
grep -E "control failed|passed|failed|^E " $RUN_DIR/r2_seams2.log | tail -6
start_job r3_stream2 bash -c "echo \$\$ > $RUN_DIR/r3_stream2.pid; exec env COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref /opt/venv/bin/python -m pytest $T/e2e/test_streaming.py --timeout=0 -p no:cacheprovider -q"
wait_job r3_stream2 86400
grep -E "\(final\)|passed|failed|^E " $RUN_DIR/r3_stream2.log | tail -12
start_job r3_regress bash -c "echo \$\$ > $RUN_DIR/r3_regress.pid; exec /opt/venv/bin/python -m pytest $T/pcc/test_upsample_conformer_encoder.py $T/pcc/test_conformer_encoder.py $T/pcc/test_flow.py $T/pcc/test_flow_checkpoint.py $T/pcc/test_flow_decoder.py --timeout=0 -p no:cacheprovider -q"
wait_job r3_regress 86400
grep -E "passed|failed|^FAILED|^E " $RUN_DIR/r3_regress.log | tail -6
