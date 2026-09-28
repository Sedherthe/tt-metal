#!/bin/bash
# Phase E1: Stage 1 under the protocol, on the kernel cache phase B built (this process allocates as those did).
source /tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0928
SCR=/tmp/claude-1000/-home-user-tt-metal/5e0d25e5-2b87-412d-8d72-f567c348a996/scratchpad
export TT_METAL_CACHE=$RUN_DIR/kcache
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
cd /home/user/tt-metal
elfs() { find $TT_METAL_CACHE -name "*.elf" | wc -l; }
DEMO="python models/experimental/cosyvoice2/demo/demo.py --inputs /home/user/data/cosyvoice2_inputs"

n0=$(elfs)
start_job stage1_bucketed bash -c "source python_env/bin/activate && $DEMO --out $RUN_DIR/stage1_bucketed --warmup buckets"
wait_job stage1_bucketed 10800 || exit 1
n1=$(elfs); echo "binaries compiled by the warmed run: $((n1 - n0))"
tail -12 $RUN_DIR/stage1_bucketed.log

FIRST=zero_shot_121-127105-0003
start_job stage1_cold_first bash -c "source python_env/bin/activate && $DEMO --out $RUN_DIR/stage1_cold_first --warmup none --cases $FIRST"
wait_job stage1_cold_first 10800 || exit 1
n2=$(elfs); echo "binaries compiled by the cold first request: $((n2 - n1))"
tail -6 $RUN_DIR/stage1_cold_first.log

# the listening pairs (device) and the WER/SIM re-score (CPU) do not time anything, so they run together
CASES=$(python3 - <<'PY'
import json
r = json.load(open("/home/user/data/cosyvoice2_runs/0928/stage1_bucketed/results.json"))["results"]
b = [128 * k for k in range(1, 9)] + [1280, 1536, 1792, 2048]
pad = sorted((next(x for x in b if x >= 2 * len(c["segment_tokens"][0])) - 2 * len(c["segment_tokens"][0]), c["case_id"]) for c in r)
pad = [p for p in pad if p[0] > 0]  # an exact fit has no padding: its pair would be identical
print(",".join(dict.fromkeys([pad[-1][1], pad[len(pad) // 2][1], pad[0][1]])))
PY
)
echo "listening cases (most, median, least padding): $CASES"
start_job listening bash -c "source python_env/bin/activate && python $SCR/listening_pairs.py $RUN_DIR/stage1_bucketed $CASES /home/user/listening"
start_job score_stage1 bash -c "cd /tmp && /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir $RUN_DIR/stage1_bucketed --baseline /home/user/data/cosyvoice2_runs/reference"
wait_job listening 5400; wait_job score_stage1 7200
cat /home/user/listening/pairs.json 2>/dev/null
grep -vE "Loading|it/s|iB/s|Warning" $RUN_DIR/score_stage1.log | tail -25
