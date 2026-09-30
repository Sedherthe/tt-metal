"""Stage 3 plan, step 2: one CFM Euler step on the device.

The reported configuration runs the CFM eager (`CosyVoice2Config.cfm_trace = False`). Per Euler step,
`TtCausalConditionalCFM._solve_eager`:
1. the host CFG-doubles x to `[2, T, 80]` and uploads it (bf16, tile layout);
2. the host builds the time embedding's raw sinusoid for t and uploads it;
3. the device runs the estimator at batch 2 (`_forward_from_raw_temb`);
4. the host downloads dphi `[2, T, 80]`, blends the CFG pair and takes the Euler update.

The conditioning comes from real chunks: the CFM's inputs are captured from the flow's own calls on
121-127105-0015's tokens (the Stage 1 demo's), at each geometry a streamed utterance meets.
- `stream_256`: its first chunk (prompt 168 + 32 + 3 look-ahead tokens -> flow bucket 256, 512 mel frames);
- `stream_384`, `stream_512`: a 100-token hop and a later chunk (121-127105-0024's tokens);
- `final_320`: 0015's final chunk, the non-streaming flow over every token, bucket 320, padded.

Per geometry: one compile solve, then R timed eager solves.
- **The whole solve** is timed as `_solve_eager` runs it.
- **One step split into its parts**, with a device sync between them. The estimator is split again into the host's
  time to enqueue it (return from the call) and the wait for the device after that.
- **Traced** (streaming geometries only; `_forward_traced`): the step captured once, then each replay timed.

`--profile` instead runs one geometry under the device profiler: a compile solve, then exactly one eager Euler step
between signposts. The ops report (`python -m tracy -r ...`) then has that step's ops with their device kernel
times; `cfm_profile_report.py` summarizes them.

    python cfm_step_profile.py --out <json> [--repeats 5]
    python -m tracy -r -p -v cfm_step_profile.py --profile stream_256 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

import ttnn

INPUTS = "/home/user/data/cosyvoice2_inputs"
TOKENS_FROM = "/home/user/data/cosyvoice2_runs/0930/stage1_masked/results.json"
PRE_LOOKAHEAD = 3


def build_flow(device):
    from huggingface_hub import hf_hub_download

    from models.experimental.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec
    from models.experimental.cosyvoice2.tt.geometry_cache import threshold_override
    from models.experimental.cosyvoice2.tt.hifigan.conv import config_tensors_in_dram_override
    from models.experimental.cosyvoice2.tt.text import MODEL_REPO_ID, MODEL_REVISION

    with config_tensors_in_dram_override(True), threshold_override(0):  # as CosyVoice2TTNN builds it
        path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="flow.pt", revision=MODEL_REVISION)
        ref = CausalMaskedDiffWithXvecRef.from_checkpoint(torch.load(path, map_location="cpu"))
        ref.eval()
        flow = TtCausalMaskedDiffWithXvec(device, ref, dtype=ttnn.bfloat16)
    flow.decoder.use_trace = False
    flow.encoder.use_trace = False
    return flow


def capture_conditioning(flow, geometries):
    """Run the flow's own call for each geometry and keep what it hands the CFM (host tensors)."""
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, bucket_for
    from models.experimental.cosyvoice2.tt.prompt import PromptContext

    results = {r["case_id"]: r for r in json.load(open(TOKENS_FROM))["results"]}
    buckets = CosyVoice2Config.reported().flow_token_buckets()
    captured, real_forward = {}, flow.decoder.forward

    def grab(mu, mask, n_timesteps, spks, cond, use_trace=None, streaming=False):
        captured["args"] = (mu.clone(), mask.clone(), spks.clone(), cond.clone(), streaming)
        return real_forward(mu, mask, n_timesteps, spks, cond, use_trace=use_trace, streaming=streaming)

    flow.decoder.forward = grab
    out = {}
    try:
        for name, (case, n, streaming) in geometries.items():
            ctx = PromptContext.from_npz(f"{INPUTS}/{case}.npz")
            tokens = [t for seg in results[case]["segment_tokens"] for t in seg]
            tok = torch.tensor([tokens[:n]], dtype=torch.int32)
            if streaming:
                bucket = bucket_for(ctx.n_prompt_tokens + n, buckets)
                flow.inference_streaming(tok, ctx.flow_prompt_speech_tokens, ctx.prompt_feat, ctx.embedding, bucket,
                                         context_len=PRE_LOOKAHEAD)  # fmt: skip
            else:
                bucket = bucket_for(ctx.n_prompt_tokens + n, buckets)
                flow.inference(tok, ctx.flow_prompt_speech_tokens, ctx.prompt_feat, ctx.embedding, bucket)
            mu, mask, spks, cond, st = captured.pop("args")
            out[name] = {"case": case, "tokens": n, "bucket": bucket, "t_len": int(mu.shape[1]),
                         "valid_frames": int(mask.sum()), "streaming": st, "args": (mu, mask, spks, cond)}  # fmt: skip
            print(f"{name}: {case} {n} tokens (+ prompt {ctx.n_prompt_tokens}) -> bucket {bucket}, T {mu.shape[1]}, "
                  f"valid {int(mask.sum())}, streaming {st}", flush=True)  # fmt: skip
    finally:
        flow.decoder.forward = real_forward
    return out


def upload_conditioning(cfm, mu, mask, spks, cond):
    """`TtCausalConditionalCFM.forward`'s own upload: CFG-doubled, DRAM, bf16."""

    def up(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=cfm.device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    return (up(torch.cat([mu, torch.zeros_like(mu)], 0)), up(torch.cat([spks, torch.zeros_like(spks)], 0).unsqueeze(1)),
            up(torch.cat([cond, torch.zeros_like(cond)], 0)), up(torch.cat([mask, mask], 0)))  # fmt: skip


def t_span_for(cfm, n_timesteps, dtype):
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=dtype)
    if cfm.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    return t_span


def chunk_bias_for(cfm, t_len, streaming, padded):
    if streaming:
        return cfm.estimator.chunk_bias_device(t_len)
    if padded:
        return ttnn.zeros((1, 1, t_len, t_len), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=cfm.device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    return None


def one_step(cfm, x, t, dt, dev, t_len, chunk_bias, parts):
    """One Euler step as `_solve_eager` takes it, each part timed with a device sync after it."""
    device, est = cfm.device, cfm.estimator
    mu_dev, spks_dev, cond_dev, mask_dev = dev
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    x_in, t_in = torch.cat([x, x], dim=0), t.repeat(2)
    x_dev = ttnn.from_torch(x_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    temb_raw = est._sinusoidal_pos_emb(t_in)
    ttnn.synchronize_device(device)
    t2 = time.perf_counter()
    dphi_dev = est._forward_from_raw_temb(x_dev, mask_dev, mu_dev, temb_raw, spks_dev, cond_dev, t_len, batch_size=2,
                                          chunk_bias=chunk_bias)  # fmt: skip
    t3 = time.perf_counter()
    ttnn.synchronize_device(device)
    t4 = time.perf_counter()
    dphi = ttnn.to_torch(dphi_dev).float()
    t5 = time.perf_counter()
    d, cfg_d = dphi[:1], dphi[1:2]
    d = (1.0 + cfm.inference_cfg_rate) * d - cfm.inference_cfg_rate * cfg_d
    x = x + dt * d.to(x.dtype)
    t6 = time.perf_counter()
    for tensor in (x_dev, temb_raw, dphi_dev):
        ttnn.deallocate(tensor)
    if parts is not None:
        for k, v in (("upload_x", t1 - t0), ("temb", t2 - t1), ("estimator_enqueue", t3 - t2),
                     ("estimator_device_wait", t4 - t3), ("download_dphi", t5 - t4), ("host_update", t6 - t5),
                     ("step", t6 - t0)):  # fmt: skip
            parts.setdefault(k, []).append(v)
    return x


def solve_parts(cfm, args, n_timesteps, parts):
    mu, mask, spks, cond = args
    t_len = mu.shape[1]
    streaming = parts.pop("_streaming")
    padded = not streaming and not bool((mask != 0).all())
    dev = upload_conditioning(cfm, mu, mask, spks, cond)
    chunk_bias = chunk_bias_for(cfm, t_len, streaming, padded)
    t_span = t_span_for(cfm, n_timesteps, mu.dtype)
    x, t, dt = cfm.rand_noise[:, :t_len, :].to(mu.dtype), t_span[0].unsqueeze(0), t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        x = one_step(cfm, x, t, dt, dev, t_len, chunk_bias, parts)
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    for tensor in (*dev, *([chunk_bias] if chunk_bias is not None else [])):
        ttnn.deallocate(tensor)
    parts["_streaming"] = streaming
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--profile", default=None, help="one geometry, one eager step between signposts")
    args = ap.parse_args()
    geometries = {
        "stream_256": ("zero_shot_121-127105-0015", 32 + PRE_LOOKAHEAD, True),
        "stream_384": ("zero_shot_121-127105-0024", 182 + PRE_LOOKAHEAD, True),
        "stream_512": ("zero_shot_121-127105-0024", 282 + PRE_LOOKAHEAD, True),
        "final_320": ("zero_shot_121-127105-0015", 95, False),
    }
    if args.profile:
        geometries = {args.profile: geometries[args.profile]}
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=200_000_000)
    report = {"repeats": args.repeats, "geometries": {}}
    try:
        flow = build_flow(device)
        cfm = flow.decoder
        t0 = time.perf_counter()
        cond = capture_conditioning(flow, geometries)  # also the compile solve of each geometry
        report["compile_s"] = round(time.perf_counter() - t0, 1)
        if args.profile:
            ttnn.ReadDeviceProfiler(device)  # flush what the compile solve left in the profiler's buffers
        if args.profile:
            from tracy import signpost

            g = cond[args.profile]
            mu, mask, spks, c = g["args"]
            parts = {"_streaming": g["streaming"]}
            solve_parts(cfm, g["args"], 10, parts)  # every program of the step, compiled and cached
            ttnn.ReadDeviceProfiler(device)
            dev = upload_conditioning(cfm, mu, mask, spks, c)
            t_len = mu.shape[1]
            padded = not g["streaming"] and not bool((mask != 0).all())
            chunk_bias = chunk_bias_for(cfm, t_len, g["streaming"], padded)
            t_span = t_span_for(cfm, 10, mu.dtype)
            x = cfm.rand_noise[:, :t_len, :].to(mu.dtype)
            ttnn.synchronize_device(device)
            signpost("euler_step_start")
            one_step(cfm, x, t_span[0].unsqueeze(0), t_span[1] - t_span[0], dev, t_len, chunk_bias, None)
            signpost("euler_step_end")
            ttnn.ReadDeviceProfiler(device)
            report["geometries"][args.profile] = {k: v for k, v in g.items() if k != "args"}
        else:
            for name, g in cond.items():
                row = {k: v for k, v in g.items() if k != "args"}
                mu, mask, spks, c = g["args"]
                # the whole solve as the pipeline runs it (eager)
                solves = []
                for _ in range(args.repeats):
                    ttnn.synchronize_device(device)
                    t0 = time.perf_counter()
                    cfm.forward(mu, mask, 10, spks, c, use_trace=False, streaming=g["streaming"])
                    solves.append(time.perf_counter() - t0)
                row["eager_solve_s"] = [round(s, 4) for s in solves]
                # one step's parts (synced between parts), every step of R solves
                parts = {"_streaming": g["streaming"]}
                for _ in range(args.repeats):
                    solve_parts(cfm, g["args"], 10, parts)
                parts.pop("_streaming")
                row["parts_ms_median"] = {k: round(1000 * float(np.median(v)), 2) for k, v in parts.items()}
                row["parts_ms_min"] = {k: round(1000 * float(np.min(v)), 2) for k, v in parts.items()}
                # traced: streaming geometries only (the traced solve refuses a padded non-streaming mask)
                if g["streaming"]:
                    replays, real_execute = [], ttnn.execute_trace

                    def timed_execute(*a, **kw):
                        t0 = time.perf_counter()
                        out = real_execute(*a, **kw)
                        replays.append(time.perf_counter() - t0)
                        return out

                    ttnn.execute_trace = timed_execute
                    try:
                        traced = []
                        for i in range(args.repeats + 1):  # the first captures
                            ttnn.synchronize_device(device)
                            t0 = time.perf_counter()
                            cfm.forward(mu, mask, 10, spks, c, use_trace=True, streaming=True)
                            traced.append(time.perf_counter() - t0)
                            if i == 0:
                                replays.clear()
                    finally:
                        ttnn.execute_trace = real_execute
                        cfm.release_cfm_trace()
                    row["traced_capture_solve_s"] = round(traced[0], 3)
                    row["traced_solve_s"] = [round(s, 4) for s in traced[1:]]
                    row["traced_replay_ms_median"] = round(1000 * float(np.median(replays)), 2)
                    row["traced_replay_ms_min"] = round(1000 * float(np.min(replays)), 2)
                report["geometries"][name] = row
                print(json.dumps({name: row}), flush=True)
    finally:
        ttnn.close_device(device)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
