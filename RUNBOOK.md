# CosyVoice2 bring-up — runbook (new pod, running, safety)

Written for the 09-29 pod (`app-5ddf2d9d-deployment-5545b7c7f8-c4vpz`). Where an older pod differed, it says so.

## 1. New pod: check the card before anything else

- **Driver and power:** `cat /sys/module/tenstorrent/version`, and `ls /sys/module/tenstorrent/parameters/`.
  - The 09-27/28 pod ran KMD 2.3.0. There, `power_policy` and `idle_power_down_grace_ms` don't exist.
  - The 09-29 pod runs **KMD 2.9.0**, the driver both dead pods ran (O2). Its parameters: `power_policy=Y`,
    `idle_power_down_grace_ms=5000`, `auto_reset_timeout=10`, `reset_limit=10`. What to do if the card drops: D36
    and §4.
- **Links:**
  ```bash
  for d in /sys/bus/pci/devices/*; do
    [ "$(cat $d/vendor)" = 0x1e52 ] && echo "$d $(cat $d/current_link_speed) x$(cat $d/current_link_width)"
  done
  ```
  - Expect 16.0 GT/s x16.
  - To map the card this container uses, read `ls -l /dev/tenstorrent/` together with
    `/sys/class/tenstorrent/*/device`. On 09-29 that was `/dev/tenstorrent/2` = `0000:01:00.0`.
- **`tt-smi -s`, twice, a few minutes apart.** Check the firmware version, DRAM status, and that the heartbeat
  advances between the two. There must be no `0xffffffff`.
- **Smoke test:** `scripts/2026-09-29/smoke_add.py`, a 64×64 `ttnn.add` with a `to_torch` read-back.
  - On a hang or a `0xffffffff`, stop and report.
  - Don't loop on resets. One `tt-smi -r` is acceptable if the link is still up.
- Run `tt-smi` and every device job as a background job with a sentinel (§3), never under a timeout.
- Record the hostname and card BDF in STATUS.md.

## 2. Environment (the 09-29 pod)

- **There is no `python_env`. Device-side work runs in `/opt/venv`,** the image's venv, which is already active
  (`VIRTUAL_ENV=/opt/venv`).
  - It has ttnn installed editable from `~/tt-metal` (`uv pip install -e .`, run by `~/setup_instance.sh`).
  - It also has torch 2.11.0+cpu, transformers 5.12.1, graphviz 0.21, pytest 9.0.3 and pre-commit 3.5.0.
  - The package adds one requirement, `inflect==7.5.0` (it pulls `typeguard`). It was installed here on 09-29 with
    the user's OK.
  - Use `/opt/venv/bin/python`. Where the package's docs say `python_env`, this venv plays that role here.
  - Older pods had a `python_env` made by `./create_venv.sh`. Don't create one without asking.
- **`~/.bashrc` on this pod:**
  - it sources `$PYTHON_ENV_DIR/bin/activate`, which is missing here, so every shell prints a harmless error;
  - it exports `PYTHONPATH=$TT_METAL_HOME`, added by `~/setup_instance.sh`. The device side doesn't need it: the
    editable install's `.pth` files already put the repo on `sys.path`.
- **Reference jobs run with `env -u PYTHONPATH`.**
  - This is a precaution. With the tt-metal root on the path, its top-level packages (`models`, `tools`, `tests`,
    …) could shadow an import on the reference side.
  - The 09-27/28 reference outputs were produced with no `PYTHONPATH`. Unsetting it reproduces that condition, and
    the regenerated reference matches (B21).
- **`TT_METAL_HOME=/home/user/tt-metal`, `HF_HOME=/home/user/models`.** Both are set by `~/.bashrc` here.
- **The checkpoint:** `FunAudioLLM/CosyVoice2-0.5B` at revision `eec1ae6c79877dbd9379285cf8789c9e0879293d`,
  4.6 GB in `$HF_HOME/hub`. It is pinned in the code (D35).
  - `snapshot_download(..., local_files_only=True)` with no revision resolves `refs/main` in the cache.
  - A snapshot downloaded by commit hash alone has no `refs/main`, so pass the revision. On 09-29, one plain
    `snapshot_download` wrote the ref.
- **The reference venv:** `~/cosyvoice2_ref_env`.
  - Build it with `docs/security.md`'s two steps: torch alone from the CPU index, then PyPI. Step 2 goes under the
    lock file (D35).
  - Check that `uv pip show torch torchaudio` says `+cpu`, and that no `nvidia-*` package is present. `triton`
    comes with openai-whisper; that's expected.
- **Upstream:** `~/CosyVoice` is `FunAudioLLM/CosyVoice` at `074ca6dc9e80`, cloned with `--recursive`
  (Matcha-TTS at `dd9105b`).
- **Data** (`/home/user/data`, rebuilt by `scripts/2026-09-29/phase_ref.sh`):

  | path | contents |
  |---|---|
  | `LibriSpeech/test-clean` | openslr 12, md5 `32fa31d27d2e1cad72775fee3f4849a9` |
  | `cosyvoice2_inputs` | 27 cases: `index.json` and `index_extension.json` |
  | `cosyvoice2_runs/reference` | the PyTorch reference, 7 cases, scored |
  | `cosyvoice2_runs/reference_tfx` | the token-accuracy extension's reference, 20 cases |
  | `cosyvoice2_token_accuracy` | 27 top-5 files |
  | `cosyvoice2_hift_stream_ref` | 3 mels for the seam gate |

  Run logs go under `cosyvoice2_runs/<MMDD>`.
- **The build:** `build/` → `build_Release`, from `build_metal.sh` (the setup script). Rebuild only if C++ changed
  since the build; compare mtimes.
- **Git:**
  - A fresh clone has no identity. Set the repo-local one: `sedherthe <siddhartha.soma1@gmail.com>`.
  - `gh` is logged in on this pod, but pushes go only to the backup branches (D34).
  - The pre-commit hook is installed for the PR repo. The worktrees share it, so notes commits need
    `PRE_COMMIT_ALLOW_NO_CONFIG=1`.
  - The first PR commit downloads the hook environments.
- Install nothing without asking. torch comes only from the CPU index.

## 3. Running

- **Job control:** `scripts/2026-09-28/jobs.sh` (`start_job` / `wait_job`, sentinel files; D25).
  - Each device job writes its pid to `<name>.pid`, for SIGINT only.
  - `scripts/2026-09-29/watch_chains.sh` streams a chain's job lines.
- **The suite:** on 09-29 it took 1 h 13 min on a cold kernel cache; on 09-28 it took 15:39 on a warm one. The perf
  test is deselected here and runs in its own process afterwards:
  ```bash
  HF_HOME=/home/user/models COSYVOICE2_INPUTS=/home/user/data/cosyvoice2_inputs \
  COSYVOICE2_TOKEN_REF=/home/user/data/cosyvoice2_token_accuracy \
  COSYVOICE2_HIFT_STREAM_REF=/home/user/data/cosyvoice2_hift_stream_ref \
  /opt/venv/bin/python -m pytest models/experimental/cosyvoice2/tests --timeout=0 -p no:cacheprovider -q -rs \
    --deselect 'models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances[device_params0]'
  ```
  `pytest.ini`'s addopts already carry `-vvs -rA`, so each test's printed figures land in the log.
- **The perf test:** the same environment, then
  `/opt/venv/bin/python -m pytest '<the deselected id>' --timeout=0 -p no:cacheprovider -q -s -rs`.
- **The Stage 1 demo:**
  `/opt/venv/bin/python models/experimental/cosyvoice2/demo/demo.py --inputs /home/user/data/cosyvoice2_inputs --out <dir>`.
  - Give it `LIBRISPEECH_ROOT=/home/user/data`, so `results.json` carries the prompt wav paths.
  - Score it from `/tmp`:
    `env -u PYTHONPATH HF_HOME=/home/user/models ~/cosyvoice2_ref_env/bin/python ~/tt-metal/models/experimental/cosyvoice2/scripts/eval_wer_sim.py --run-dir <dir> --baseline /home/user/data/cosyvoice2_runs/reference`.
- **The opt-in allocation tracker** must be the only test in its invocation:
  `COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1 /opt/venv/bin/python -m pytest models/experimental/cosyvoice2/tests/pcc/test_flow_decoder.py::test_device_cfm_traces_pass_allocation_tracker -s -q`
- **Scripts:** `/opt/venv/bin/python <script>`. Some scripts find the repo through `parents[N]`, so run them from
  their in-repo path. The dated scripts under `scripts/perf_*` and `vocoder_debug_*` still name the old
  `models/demos/audio/cosyvoice2` location; the package has been at `models/experimental/cosyvoice2` since D19.
- **Environment flags**, read at construction, with their defaults:
  - `COSYVOICE2_FLOW_SDPA=1`
  - `COSYVOICE2_FLOW_FUSED_QKV=1`
  - `COSYVOICE2_FLOW_CFM_TRACE=0`
  - `COSYVOICE2_FLOW_ENCODER_TRACE=0`
  - `COSYVOICE2_CONV_CONFIG_IN_DRAM=1`
  - `COSYVOICE2_CFM_TRACE_CACHE_CAPACITY=1` (only 1 is accepted)

  The LLM's `use_decode_trace` is off by default.

## 4. Safety rules

- **No `timeout` wrapper on device jobs.** If a run looks stuck, tell the user. Stop only with SIGINT, **never
  SIGKILL**: it can wedge the card. Many older scripts' docstrings still say `timeout -s KILL`; ignore them.
- **If the card drops** (a PCIe error, `0xffffffff`, a hang):
  - stop all device work;
  - don't retry resets in a loop;
  - make sure the backup branches hold every commit;
  - report the exact error and what was running (D36).
- `ttnn.close_device` keeps UMD's `CHIP_IN_USE` lock until process exit. A second process that opens the card
  blocks, and it ignores SIGINT while blocked.
- Don't `pkill -f` a pattern that can match your own shell.
- **Traces:**
  - A trace kept alive across the flow and the vocoder hung the card once (09-21).
  - The LLM decode trace is scoped to one `generate()`.

## 5. Branches

- **`bringup/cosyvoice2-istft`:** the PR branch (draft PR #56651). Commit only as authorized (D34). The user
  pushes it.
- **`notes/cosyvoice2`:** this branch. It's an orphan and is never merged. The user pushes it.
  - A fresh clone has no local branch. Make the worktree with
    `git worktree add --track -b notes/cosyvoice2 ~/cosyvoice2-notes-wt origin/notes/cosyvoice2`.
- **`backup/<date>-pr`, `backup/<date>-notes`:** every commit is pushed there at once, and its hash reported (D34).
- **`wip/cfm-streaming-2026-09-25`:** a file source only. Don't commit to it again.
