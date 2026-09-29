#!/bin/bash
# D35 check: the pinned revision resolves the same snapshot on both sides, and the reference side's outputs don't move.
set -e
M=/home/user/tt-metal/models/experimental/cosyvoice2
S=/tmp/claude-1000/-home-user-tt-metal/5b576035-d516-4f80-869b-0f1cc86dd847/scratchpad
REF="env -u PYTHONPATH OMP_NUM_THREADS=48 COSYVOICE2_REPO=/home/user/CosyVoice HF_HOME=/home/user/models LIBRISPEECH_ROOT=/home/user/data /home/user/cosyvoice2_ref_env/bin/python"
cd /tmp
echo "== reference side: model_dir()"
$REF -c "import sys; sys.path.insert(0, '$M/scripts'); import reference_env as r; print(r.MODEL_REVISION, r.model_dir())"
echo "== reference side: shim self-test"
$REF $M/tests/reference/test_reference_env.py 2>&1 | grep -E 'PASSED|FAILED|Error'
echo "== reference side: prepare_inputs.py under the pin vs today's inputs"
rm -rf $S/pin_inputs
$REF $M/scripts/prepare_inputs.py --parity --out-dir $S/pin_inputs | tail -1
python3 - <<EOF
import glob, os, numpy as np
new_dir, old_dir = "$S/pin_inputs", "/home/user/data/cosyvoice2_inputs"
n = 0
for p in sorted(glob.glob(os.path.join(new_dir, "*.npz"))):
    a, b = np.load(p), np.load(os.path.join(old_dir, os.path.basename(p)))
    assert set(a.files) == set(b.files), p
    for k in a.files:
        assert np.array_equal(a[k], b[k]), (p, k)
    n += 1
print(f"{n} cases: every array bit-identical to /home/user/data/cosyvoice2_inputs")
EOF
rm -rf $S/pin_inputs
echo "== device side: the pinned downloads resolve into the same snapshot"
cd /home/user/tt-metal && HF_HOME=/home/user/models /opt/venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
from models.experimental.cosyvoice2.tt.text import MODEL_REPO_ID, MODEL_REVISION, tokenizer_dir
for f in ("llm.pt", "flow.pt", "hift.pt", "CosyVoice-BlankEN/config.json"):
    print(f, hf_hub_download(repo_id=MODEL_REPO_ID, filename=f, revision=MODEL_REVISION))
print("tokenizer", tokenizer_dir())
EOF
