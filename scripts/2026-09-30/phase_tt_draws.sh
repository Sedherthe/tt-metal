#!/bin/bash
# D43's TT side on the masked HiFT (ed1c3ad1c5): five vocoder noise draws (seeds 1-5) over the demo's tokens, Stage 1
# and streaming (scripts/noise_draws.py, one process, the demo's warm-ups). Then all four groups scored and set side
# by side with the reference draws (scripts/eval_draws.py; the reference's scores are reused).
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/0930
KC=/home/user/.cache/tt-metal-cache
D=/home/user/data/cosyvoice2_draws
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
cd /home/user/tt-metal
n0=$(find $KC -name "*.elf" | wc -l)
start_job tt_draws bash -c "echo \$\$ > $RUN_DIR/tt_draws.pid; exec /opt/venv/bin/python models/experimental/cosyvoice2/scripts/noise_draws.py --inputs /home/user/data/cosyvoice2_inputs --out $D"
wait_job tt_draws 86400 || { grep -E 'Error|Traceback' -A3 $RUN_DIR/tt_draws.log | tail -12; exit 1; }
echo "binaries compiled by tt_draws: $(( $(find $KC -name '*.elf' | wc -l) - n0 ))"
tail -1 $RUN_DIR/tt_draws.log
g() { echo $(for s in 1 2 3 4 5; do echo "$D/$1_seed$s"; done); }
cd /tmp
start_job score_draws bash -c "exec env -u PYTHONPATH OMP_NUM_THREADS=48 LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models /home/user/cosyvoice2_ref_env/bin/python /home/user/tt-metal/models/experimental/cosyvoice2/scripts/eval_draws.py --reuse --out $RUN_DIR/draws.json --group 'TT Stage 1' $(g tt_stage1) --group 'reference Stage 1' $(g ref_stage1) --group 'TT streaming' $(g tt_stream) --group 'reference streaming' $(g ref_stream)"
wait_job score_draws 14400
sed -n '/WER % and SIM/,$p' $RUN_DIR/score_draws.log
