#!/bin/bash
# 2026-10-01: restore the environment on the new pod (CPU and network only), the jobs in parallel. RUNBOOK section 2.
#  - inflect 7.5.0 into /opt/venv (the package's one requirement; the user's OK, 10-01)
#  - the reference venv, docs/security.md's two steps, step 2 under the lock file (D35)
#  - upstream CosyVoice at 074ca6dc9e80, recursive
#  - the checkpoint at the pinned revision only (no plain snapshot_download: main may have moved), and
#    Qwen/Qwen2-0.5B-Instruct for the LLM tests (unpinned there; its revision is logged)
#  - LibriSpeech test-clean, md5-checked
source /home/user/cosyvoice2-notes-wt/scripts/2026-09-28/jobs.sh
RUN_DIR=/home/user/data/cosyvoice2_runs/1001
P=/home/user/tt-metal/models/experimental/cosyvoice2
ENV=/home/user/cosyvoice2_ref_env
cd /tmp
for d in $ENV /home/user/CosyVoice /home/user/data/LibriSpeech; do [ -e "$d" ] && { echo "$d exists; refusing"; exit 2; }; done

start_job inflect uv pip install --python /opt/venv/bin/python inflect==7.5.0
start_job ref_venv bash -c "set -e
  uv venv --python 3.10 $ENV
  VIRTUAL_ENV=$ENV uv pip install -r $P/requirements-reference-torch.txt
  VIRTUAL_ENV=$ENV uv pip install -r $P/requirements-reference.txt -c $P/requirements-reference-lock.txt
  uv pip show --python $ENV/bin/python torch torchaudio | grep -E '^(Name|Version)'
  echo nvidia packages: \$(uv pip list --python $ENV/bin/python | grep -ci nvidia || true)
  uv pip freeze --python $ENV/bin/python > $RUN_DIR/ref_env_freeze.txt
  echo frozen: \$(wc -l < $RUN_DIR/ref_env_freeze.txt) packages"
start_job upstream bash -c "set -e
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /home/user/CosyVoice
  git -C /home/user/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
  git -C /home/user/CosyVoice submodule update --init --recursive
  git -C /home/user/CosyVoice log -1 --format='upstream %H %ci'
  git -C /home/user/CosyVoice submodule status --recursive"
start_job checkpoints bash -c "set -e
  env -u PYTHONPATH HF_HOME=/home/user/models /opt/venv/bin/python -c \"
from huggingface_hub import snapshot_download as s, HfApi
p = s('FunAudioLLM/CosyVoice2-0.5B', revision='eec1ae6c79877dbd9379285cf8789c9e0879293d'); print('CosyVoice2-0.5B', p)
q = s('Qwen/Qwen2-0.5B-Instruct'); print('Qwen2-0.5B-Instruct', q)
\"
  du -sh /home/user/models/hub/*"
start_job librispeech bash -c "set -e
  mkdir -p /home/user/data
  curl -sSL -o /home/user/data/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
  md5=\$(md5sum /home/user/data/test-clean.tar.gz | cut -d' ' -f1); echo md5 \$md5
  [ \$md5 = 32fa31d27d2e1cad72775fee3f4849a9 ] || { echo md5 mismatch; exit 1; }
  tar xzf /home/user/data/test-clean.tar.gz -C /home/user/data
  ls /home/user/data/LibriSpeech/test-clean | wc -l"
rc=0
for j in inflect ref_venv upstream checkpoints librispeech; do wait_job $j 14400 || rc=1; done
echo "env chain done, rc=$rc"
exit $rc
