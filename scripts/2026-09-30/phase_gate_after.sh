#!/bin/bash
# D41's gates on the masked HiFT (they failed on the old padding: phase_gate_before.sh), then the rest of the streaming
# tests (stage B's warm-ups now take the masked path) and the HiFT/F0 tests, for regressions.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
export COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref
T=models/experimental/cosyvoice2/tests
cd /home/user/tt-metal
n0=$(find $KC -name "*.elf" | wc -l)
start_job gate_after bash -c "echo \$\$ > $RUN_DIR/gate_after.pid; exec /opt/venv/bin/python -m pytest $T/pcc/test_hift_masked.py $T/e2e/test_streaming.py --timeout=0 -p no:cacheprovider -q -s"
wait_job gate_after 86400
echo "binaries compiled by gate_after: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
grep -E "^\| zero|^E |passed|failed|chunks,|chunk offset" $RUN_DIR/gate_after.log | cut -c1-260 | tail -45
n0=$(find $KC -name "*.elf" | wc -l)
start_job hift_regress bash -c "echo \$\$ > $RUN_DIR/hift_regress.pid; exec /opt/venv/bin/python -m pytest $T/pcc/test_hift_chunked.py $T/pcc/test_hift_checkpoint.py $T/pcc/test_hift_generator_inference.py $T/pcc/test_hift_decode.py $T/pcc/test_f0_predictor.py $T/pcc/test_f0_predictor_checkpoint.py $T/pcc/test_istft.py $T/pcc/test_stft.py $T/pcc/test_sine_gen2.py $T/pcc/test_flow_to_hift.py --timeout=0 -p no:cacheprovider -q -rs"
wait_job hift_regress 86400
echo "binaries compiled by hift_regress: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
grep -E "^E |passed|failed|SKIPPED" $RUN_DIR/hift_regress.log | cut -c1-240 | tail -12
