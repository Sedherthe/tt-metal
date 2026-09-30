#!/bin/bash
# CosyVoice2Config.flow_n_timesteps, end to end: noise_draws.py --flow-steps 5, noise seed 1, must reproduce the step
# sweep's 5-step draw (steps_draws.py set the module constant instead), Stage 1 and streaming, wav for wav.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
OUT=$RUN_DIR/flow_steps5_option
export HF_HOME=/home/user/models
cd /home/user/tt-metal
rm -rf $OUT
start_job flow_steps_option bash -c "echo \$\$ > $RUN_DIR/flow_steps_option.pid; exec /opt/venv/bin/python models/experimental/cosyvoice2/scripts/noise_draws.py --inputs /home/user/data/cosyvoice2_inputs --out $OUT --noise-seeds 1 --flow-steps 5"
wait_job flow_steps_option 86400 || { grep -E 'Error|Traceback' -A3 $RUN_DIR/flow_steps_option.log | tail -12; exit 1; }
python3 - <<PY
import glob, os, soundfile, numpy as np
for mode in ("stage1", "stream"):
    a, b = "$OUT/tt_%s_seed1" % mode, "/home/user/data/cosyvoice2_steps/steps5/tt_%s_seed1" % mode
    same = [np.array_equal(soundfile.read(p, dtype="int16")[0], soundfile.read(os.path.join(b, os.path.basename(p)), dtype="int16")[0])
            for p in sorted(glob.glob(a + "/*.wav"))]
    print(f"{mode}: {sum(same)} of {len(same)} wavs identical to the sweep's 5-step draw")
PY
