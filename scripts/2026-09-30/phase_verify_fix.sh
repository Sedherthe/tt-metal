#!/bin/bash
# After phase_gate_after: the fix against its reference (masked vs exact length, masked_vs_exact.py), then the "you"
# clip's regression test, both halves (the device renders 11 draws; the reference venv transcribes them).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
YOU=$RUN_DIR/you_clip_draws
cd /home/user/tt-metal
wait_job phase_gate_after 86400
n0=$(find $KC -name "*.elf" | wc -l)
start_job masked_vs_exact bash -c "echo \$\$ > $RUN_DIR/masked_vs_exact.pid; exec /opt/venv/bin/python /home/user/cosyvoice2-notes-wt/scripts/2026-09-30/masked_vs_exact.py --out $RUN_DIR/masked_vs_exact"
wait_job masked_vs_exact 86400 || { grep -E 'Error|Traceback' -A3 $RUN_DIR/masked_vs_exact.log | tail -12; exit 1; }
echo "binaries compiled by masked_vs_exact: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
sed -n '/^| case | path/,$p' $RUN_DIR/masked_vs_exact.log
rm -rf $YOU
start_job you_device bash -c "echo \$\$ > $RUN_DIR/you_device.pid; exec env COSYVOICE2_YOU_OUT=$YOU /opt/venv/bin/python -m pytest models/experimental/cosyvoice2/tests/e2e/test_streaming.py::test_device_you_clip_noise_draws --timeout=0 -p no:cacheprovider -q -s"
wait_job you_device 86400
grep -E "dBFS|passed|failed|^E " $RUN_DIR/you_device.log | cut -c1-200 | tail -16
start_job you_asr_check bash -c "cd /tmp && exec env -u PYTHONPATH OMP_NUM_THREADS=48 COSYVOICE2_YOU_OUT=$YOU /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/tests/reference/test_you_clip.py"
wait_job you_asr_check 7200
grep -vE "Loading|it/s|iB/s|Warning|warn" $RUN_DIR/you_asr_check.log | tail -14
