#!/bin/bash
# After the R1 verification chain: R2's seam gate on the voiced-seam reference, then R3's streaming gate (stage A),
# whose offline-streamed wavs are scored afterwards on CPU.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0929
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
T=models/experimental/cosyvoice2/tests
cd /home/user/tt-metal
wait_job phase_r1verify 86400  # the device is free once the whole R1 chain has finished
start_job r2_seams bash -c "echo \$\$ > $RUN_DIR/r2_seams.pid; exec env COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref_r2 /opt/venv/bin/python -m pytest $T/pcc/test_hift_chunked.py --timeout=0 -p no:cacheprovider -q"
wait_job r2_seams 86400
grep -E "^\| |control failed|passed|failed" $RUN_DIR/r2_seams.log | tail -30
OUT=$RUN_DIR/stream_offline
start_job r3_stream bash -c "echo \$\$ > $RUN_DIR/r3_stream.pid; exec env COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref COSYVOICE2_STREAM_OUT=$OUT /opt/venv/bin/python -m pytest $T/e2e/test_streaming.py --timeout=0 -p no:cacheprovider -q"
wait_job r3_stream 86400
grep -E "^\| |own F0|passed|failed|Error" $RUN_DIR/r3_stream.log | tail -45
if [ -f $OUT/results.json ]; then
  start_job score_stream bash -c "cd /tmp && env -u PYTHONPATH OMP_NUM_THREADS=48 /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $OUT --baseline /home/user/data/cosyvoice2_streaming_ref"
  wait_job score_stream 7200
  grep -vE "Loading|it/s|iB/s|Warning|warn|mask" $RUN_DIR/score_stream.log | tail -14
fi
