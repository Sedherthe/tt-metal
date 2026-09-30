"""Stage 3 plan, step 3, the reference half: upstream's PyTorch CosyVoice2 at another CFM step count. RUN IN THE
REFERENCE VENV.

Upstream hardcodes `n_timesteps=10` in `CausalMaskedDiffWithXvec.inference`, which calls
`CausalConditionalCFM.forward(..., n_timesteps=10, ...)`. This replaces that forward, in this process only, with one
that passes `--steps` instead, then runs one of the PR's reference scripts unchanged (`run_reference.py` or
`streaming_reference.py`) with the arguments after `--`. The CFM's fixed noise (`rand_noise`) and its cosine time
schedule are upstream's; only the number of Euler steps changes.

    python ref_steps.py --steps 6 -- <the PR's scripts dir>/streaming_reference.py --inputs ... --noise-seed 1 ...
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("script")
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    import reference_env

    reference_env.setup_upstream()
    from cosyvoice.flow import flow_matching

    upstream_forward = flow_matching.CausalConditionalCFM.forward
    seen = []

    def forward(self, mu, mask, n_timesteps, *a, **kw):
        if not seen:
            print(f"ref_steps: CausalConditionalCFM.forward asked for {n_timesteps} steps, running {args.steps}", flush=True)
        seen.append(n_timesteps)
        return upstream_forward(self, mu, mask, args.steps, *a, **kw)

    flow_matching.CausalConditionalCFM.forward = forward
    sys.argv = [args.script] + [a for a in args.script_args if a != "--"]
    try:
        runpy.run_path(args.script, run_name="__main__")
    finally:
        print(f"ref_steps: {len(seen)} CFM calls, each run at {args.steps} steps (asked: {sorted(set(seen))})", flush=True)


if __name__ == "__main__":
    main()
