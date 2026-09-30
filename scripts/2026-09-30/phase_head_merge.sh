#!/bin/bash
# Stage 3 lever (a): the CFM's head merge by ttnn.experimental.nlp_concat_heads (tt/flow/decoder.py).
# 1. head_merge_check.py: new against committed, same inputs, four geometries (one step, the 10-step solve).
# 2. cfm_step_profile.py: eager and traced per step (against cfm_time.json), then the device profiler's step.
# 3. noise_draws.py, five draws: wav for wav against D43's (the committed merge), and the Stage 1 RTF.
# 4. The device suite, with the Stage 1 draws scored alongside on the CPU.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
S=/home/user/cosyvoice2-notes-wt/scripts/2026-09-30
D=/home/user/data/cosyvoice2_merge
D43=/home/user/data/cosyvoice2_draws
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
cd /home/user/tt-metal
start_job merge_check bash -c "echo \$\$ > $RUN_DIR/merge_check.pid; exec /opt/venv/bin/python $S/head_merge_check.py --out $RUN_DIR/merge_check.json"
wait_job merge_check 86400 || { grep -E 'Error|Traceback' -A5 $RUN_DIR/merge_check.log | tail -20; exit 1; }
grep -E '^\{"(stream|final)' $RUN_DIR/merge_check.log
start_job merge_time bash -c "echo \$\$ > $RUN_DIR/merge_time.pid; exec /opt/venv/bin/python $S/cfm_step_profile.py --out $RUN_DIR/merge_time.json --repeats 5"
wait_job merge_time 86400 || exit 1
rm -rf $RUN_DIR/merge_tracy
start_job merge_tracy bash -c "echo \$\$ > $RUN_DIR/merge_tracy.pid; exec /opt/venv/bin/python -m tracy -r -p -v -o $RUN_DIR/merge_tracy --op-support-count 5000 $S/cfm_step_profile.py --profile stream_256 --out $RUN_DIR/merge_profile.json"
wait_job merge_tracy 86400  # tracy's ops report asserts on the compile solve's dropped markers (B36); the raw logs are read
python3 $S/cfm_profile_raw.py $RUN_DIR/merge_tracy/.logs --out $RUN_DIR/merge_profile_raw.json | head -14
rm -rf $D
start_job merge_draws bash -c "echo \$\$ > $RUN_DIR/merge_draws.pid; exec /opt/venv/bin/python models/experimental/cosyvoice2/scripts/noise_draws.py --inputs /home/user/data/cosyvoice2_inputs --out $D"
wait_job merge_draws 86400 || { grep -E 'Error|Traceback' -A5 $RUN_DIR/merge_draws.log | tail -20; exit 1; }
python3 - <<PY
import glob, os, soundfile, numpy as np
for mode in ("stage1", "stream"):
    same = n = 0
    for s in range(1, 6):
        for p in sorted(glob.glob("$D/tt_%s_seed%d/*.wav" % (mode, s))):
            q = os.path.join("$D43/tt_%s_seed%d" % (mode, s), os.path.basename(p))
            same += np.array_equal(soundfile.read(p, dtype="int16")[0], soundfile.read(q, dtype="int16")[0]); n += 1
    print(f"{mode}: {same} of {n} wavs identical to D43's draws (the committed merge)")
PY
g() { echo $(for s in 1 2 3 4 5; do echo "$1_seed$s"; done); }
(cd /tmp && start_job score_merge bash -c "exec env -u PYTHONPATH OMP_NUM_THREADS=32 LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_draws.py --out $RUN_DIR/merge_scores.json --group 'TT Stage 1, merged by nlp_concat_heads' $(g $D/tt_stage1) --group 'TT Stage 1, D43 (committed merge)' $(g $D43/tt_stage1)")
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy
export COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref COSYVOICE2_STREAM_REF=/home/user/data/cosyvoice2_streaming_ref
start_job merge_suite bash -c "echo \$\$ > $RUN_DIR/merge_suite.pid; exec /opt/venv/bin/python -m pytest models/experimental/cosyvoice2/tests --timeout=0 -p no:cacheprovider -q -rs --durations=10 --deselect models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py"
wait_job merge_suite 86400
grep -E "passed|failed|^E |SKIPPED" $RUN_DIR/merge_suite.log | cut -c1-220 | tail -12
wait_job score_merge 86400
sed -n '/WER % and SIM/,$p' $RUN_DIR/score_merge.log
