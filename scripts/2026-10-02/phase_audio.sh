#!/bin/bash
# 2026-10-02: reference audio from the merged PR head (33aa3601eb; main's 98a19fd851 has the same model code), for the
# notes: the Stage 1 demo (six corpus utterances + the CosyVoice1-parity sentence) and the streaming demo (six), the
# reported configuration, seed 1986. Each run's tokens are checked against the record (B44: this card's decode is not
# always reproducible); a run whose tokens differ is run once more. Then WER/SIM in the reference venv.
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1002
S=/home/user/tt-metal/models/experimental/cosyvoice2/scripts
REFPY="env -u PYTHONPATH OMP_NUM_THREADS=48 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
export HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data COSYVOICE2_REPO=/home/user/CosyVoice
export COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs
DEMO="/opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS --warmup buckets"
RECORD=213/95/347/75/317/202
cd /home/user/tt-metal
[ "$(git rev-parse --short=10 HEAD)" = 33aa3601eb ] || { echo "HEAD is $(git rev-parse --short=10 HEAD), not 33aa3601eb"; exit 2; }
tokens() { python3 -c "
import json; r = json.load(open('$1/results.json'))['results']
print('/'.join(str(sum(len(s) for s in c['segment_tokens'])) for c in r if 'parity' not in c['case_id']))"; }
run() {  # run NAME ARGS...: the demo into $RUN_DIR/NAME, once more if its tokens are not the record's
  local name=$1; shift
  for attempt in a b; do
    start_job ${name}_$attempt bash -c "echo \$\$ > $RUN_DIR/${name}_$attempt.pid; exec $DEMO --out $RUN_DIR/${name}_$attempt $*"
    wait_job ${name}_$attempt 86400 || return 1
    t=$(tokens $RUN_DIR/${name}_$attempt); echo "$name attempt $attempt: tokens $t (record $RECORD)"
    grep -E "^Distinct" $RUN_DIR/${name}_$attempt.log | tail -1
    [ "$t" = "$RECORD" ] && { ln -sfn ${name}_$attempt $RUN_DIR/$name; return 0; }
  done
  echo "$name: tokens differ from the record in both attempts; keeping attempt b"; ln -sfn ${name}_b $RUN_DIR/$name
}
mkdir -p $RUN_DIR
run stage1 --parity || exit 1
run stream --stream || exit 1
for name in stage1 stream; do
  (cd /tmp && start_job score_$name bash -c "exec $REFPY $S/eval_wer_sim.py --run-dir $RUN_DIR/$name/ --baseline /home/user/data/cosyvoice2_runs/reference")
  wait_job score_$name 7200
  grep -A12 "^| case" $RUN_DIR/score_$name.log | head -12
done
echo "audio chain done"
