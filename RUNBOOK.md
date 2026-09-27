# CosyVoice2 bring-up — runbook (new pod, running, safety)

## 1. New pod: check the card before anything else

- Driver and power: `cat /sys/module/tenstorrent/version`, and `ls /sys/module/tenstorrent/parameters/`. On
  KMD 2.3.0 the `power_policy` and `idle_power_down_grace_ms` parameters don't exist.
- Links:
  ```bash
  for d in /sys/bus/pci/devices/*; do
    [ "$(cat $d/vendor)" = 0x1e52 ] && echo "$d $(cat $d/current_link_speed) x$(cat $d/current_link_width)"
  done
  ```
  Expect 16.0 GT/s x16. Map the card this container uses: `ls -l /dev/tenstorrent/` together with
  `/sys/class/tenstorrent/*/device`.
- `tt-smi -s`: firmware version, DRAM status, a heartbeat that advances, and no `0xffffffff`.
- Smoke test: a 64×64 `ttnn.add` with a `to_torch` read-back. Stop and report on a hang or a `0xffffffff`. Don't
  loop on resets; one `tt-smi -r` is acceptable if the link is still up.
- Record the hostname and card BDF in STATUS.md.

## 2. Environment

- `python_env` is what `~/.bashrc` activates (`$PYTHON_ENV_DIR`). If it's missing, run `./create_venv.sh`
  (Python 3.10, matching the build), **after asking the user**. Its `.pth` files put the repo, `ttnn/` and
  `tools/` on `sys.path`, so no `PYTHONPATH` is needed.
- `/opt/venv` lacks `graphviz`, so `ttnn` doesn't import there.
- Install nothing without asking. torch only via `--extra-index-url https://download.pytorch.org/whl/cpu`, then
  check that `uv pip show torch torchaudio` shows `+cpu`.
- `export TT_METAL_HOME=/home/user/tt-metal HF_HOME=/home/user/models`. Checkpoints download from
  `FunAudioLLM/CosyVoice2-0.5B` on first use.
- The build is `build/` from `build_metal.sh`. Rebuild only if C++ changed since the build (compare mtimes).
- `create_venv.sh` installs a pre-commit hook. The first commit downloads the hook environments, which takes a
  few minutes.
- git identity (repo-local): `sedherthe <siddhartha.soma1@gmail.com>`. Pods have no GitHub credentials; the user
  pushes.

## 3. Running

- Full suite on the N150 (4 min warm, about 35 min cold):
  `python_env/bin/python -m pytest models/demos/audio/cosyvoice2/tests -q -p no:cacheprovider`
- Opt-in allocation tracker (must be the only test in its invocation):
  `COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1 python_env/bin/python -m pytest models/demos/audio/cosyvoice2/tests/pcc/test_flow_decoder.py::test_device_cfm_traces_pass_allocation_tracker -s -q`
- Scripts: `python_env/bin/python <script>`. Some scripts find the repo through `parents[N]`, so run them from
  their in-repo path.
- Environment flags, read at construction, with their defaults:
  - `COSYVOICE2_FLOW_SDPA=1`
  - `COSYVOICE2_FLOW_FUSED_QKV=1`
  - `COSYVOICE2_FLOW_CFM_TRACE=0`
  - `COSYVOICE2_FLOW_ENCODER_TRACE=0`
  - `COSYVOICE2_CONV_CONFIG_IN_DRAM=1`
  - `COSYVOICE2_CFM_TRACE_CACHE_CAPACITY=1` (only 1 is accepted)

  The LLM's `use_decode_trace` is off by default.

## 4. Safety rules

- **No `timeout` wrapper on device jobs.** If a run looks stuck, tell the user. Stop only with SIGINT, **never
  SIGKILL** (it can wedge the card). Many older scripts' docstrings still say `timeout -s KILL`; ignore them.
- `ttnn.close_device` keeps UMD's `CHIP_IN_USE` lock until process exit. A second process that opens the card
  blocks, and it ignores SIGINT while blocked.
- Don't `pkill -f` a pattern that can match your own shell.
- A trace kept alive across the flow and the vocoder hung the card once (09-21). The LLM decode trace is scoped to
  one `generate()`.

## 5. Branches

- `bringup/cosyvoice2-istft`: the PR branch (draft PR #56651). Commit only when asked.
- `notes/cosyvoice2`: this branch. It's an orphan and is never merged. To edit it without touching the PR tree:
  `git worktree add ../cosyvoice2-notes notes/cosyvoice2`. The shared pre-commit hook finds no config there, so
  commit with `PRE_COMMIT_ALLOW_NO_CONFIG=1`.
- `wip/cfm-streaming-2026-09-25`: a file source only. Don't commit to it again.
