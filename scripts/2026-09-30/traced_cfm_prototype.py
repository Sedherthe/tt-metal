"""Stage 3 lever (b), evidence for a proposal and not the PR: every streaming flow bucket's CFM step traced at start-up
and alive for the whole process, alongside the LLM's decode trace, also captured once at start-up. Run under
`TT_METAL_TRACE_ALLOC_TRACKING=1` it is the proof that the traces can coexist; run without it, it is the timing.

The allocation tracker (`tt_metal/impl/allocator/trace_allocation_tracker.cpp`) flags, at every `execute_trace`,
any non-trace buffer that was allocated after that trace was captured and is still alive. So every persistent
buffer must exist before the first capture, and a capture may allocate only transient intermediates. That rules out
two things the pipeline does today:
- The CFM's capture creates its output (`next_x`) inside the capture. Here each body writes its result into a
  pre-allocated buffer instead (`ttnn.add(..., output_tensor=...)`).
- The LLM re-captures its decode trace on every `generate()` and releases it at the end, allocating its inputs and
  output each time. Here it is captured once, with its inputs pre-allocated, and `generate()`'s release is disabled.

Sequence, one process:
1. Build the pipeline (reported configuration), `warmup_buckets()`, `warmup_streaming()`.
2. Baseline: stream the corpus's six utterances eagerly (RAS seed 1986, noise seed 1).
3. Pre-allocate every CFM slot (all flow buckets, `streaming=True`) and the LLM decode trace's inputs. Compile, still
   before any capture, every program the traces and their refills use: two eager runs of each body, and each copy.
4. Capture, in `--order`: `cfm-first` (the 17 CFM traces, then the decode trace) or `llm-first`. Measure the trace
   region after each capture (`ttnn.get_memory_view(device, BufferType.TRACE)`).
5. Stream the same six again, with every streamed chunk's CFM replaying its bucket's trace. The final chunk stays
   eager: its mask is padded and non-streaming, which the traced solve does not take. The LLM decodes through its
   start-up trace.
6. Compare with 2: the tokens, and each utterance's audio (max |diff|, PCC); the tracker's verdict; first audio and RTF.

`--llm-output`: `preallocated` (the design) writes the decode trace's logits into a buffer allocated before any
capture (`ttnn.linear(..., optional_output_tensor=...)`); `inside` allocates them inside the capture, as
`_decode_step_traced` does today. That is the negative control: it must be flagged when the LLM is captured after the
CFM traces.

    TT_METAL_TRACE_ALLOC_TRACKING=1 python traced_cfm_prototype.py --order cfm-first --out <json>   # the proof
    python traced_cfm_prototype.py --order cfm-first --out <json>                                    # the timing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

import ttnn
from models.experimental.cosyvoice2.tt.flow import decoder as dec
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN
from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources
from models.tt_transformers.tt.common import Mode

INPUTS = "/home/user/data/cosyvoice2_inputs"
TRACE_REGION = 400_000_000
CH = 80


def trace_bytes(device) -> int:
    """Bytes allocated in the trace region (every bank), or -1 if the view is unavailable."""
    try:
        v = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)
    except Exception as e:  # noqa: BLE001
        print(f"trace region view unavailable: {e}", flush=True)
        return -1


def dram_free(device) -> int:
    try:
        v = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return int(v.total_bytes_free_per_bank) * int(v.num_banks)
    except Exception as e:  # noqa: BLE001
        print(f"DRAM view unavailable: {e}", flush=True)
        return -1


def dram_zeros(device, shape):
    return ttnn.from_torch(torch.zeros(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def raw_weight_convs(estimator, t_lens) -> dict:
    """The streaming buckets a trace cannot take: those where some conv's check chose the raw weight (#36487, neither
    prepared candidate right). That weight is a host tensor, so `conv1d` uploads and prepares it on every call, and a
    trace cannot contain a write. Walks the estimator for conv modules and reads their verdicts at `(t_len, 2)`."""
    convs, seen, todo = [], set(), [("estimator", estimator)]
    while todo:
        name, obj = todo.pop()
        if id(obj) in seen or obj is None or isinstance(obj, (int, float, str, bytes, torch.Tensor, ttnn.Tensor)):
            continue
        seen.add(id(obj))
        if hasattr(obj, "_verified_config") and hasattr(obj, "weight"):
            convs.append((name, obj))
        items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, (list, tuple)) else (
            vars(obj).items() if hasattr(obj, "__dict__") and type(obj).__module__.startswith("models.") else [])
        for k, v in items:
            todo.append((f"{name}.{k}", v))
    raw = {}
    for t_len in t_lens:
        for name, conv in convs:
            verdict = conv._verified_config.get((t_len, 2))
            if verdict is not None and verdict[0] is conv.weight:
                raw.setdefault(t_len, []).append(f"{name} Conv1d({conv.in_channels}->{conv.out_channels}, k={conv.kernel_size})")
    return raw


def make_slot(cfm, t_len: int) -> dec._CfmTraceSlot:
    d = cfm.device
    return dec._CfmTraceSlot(
        trace_id=None,
        next_x=dram_zeros(d, (1, t_len, CH)),  # pre-allocated: the body writes into it (output_tensor)
        x_buf=dram_zeros(d, (1, t_len, CH)),
        temb_raw_buf=dram_zeros(d, (2, 1, cfm.estimator.time_embeddings_dim)),
        dt_buf=dram_zeros(d, (1, 1, 1)),
        mu2_buf=dram_zeros(d, (2, t_len, CH)),
        spks2_buf=dram_zeros(d, (2, 1, CH)),
        cond2_buf=dram_zeros(d, (2, t_len, CH)),
        mask2_buf=dram_zeros(d, (2, t_len, 1)),
        chunk_bias_buf=cfm.estimator.chunk_bias_device(t_len),
    )


def body(cfm, slot, t_len: int) -> None:
    """`TtCausalConditionalCFM._capture`'s traced Euler step, with its result written into the pre-allocated
    `slot.next_x` rather than a tensor the capture allocates."""
    x2 = ttnn.concat([slot.x_buf, slot.x_buf], dim=0)
    d = cfm.estimator._forward_from_raw_temb(x2, slot.mask2_buf, slot.mu2_buf, slot.temb_raw_buf, slot.spks2_buf,
                                             slot.cond2_buf, t_len, batch_size=2, chunk_bias=slot.chunk_bias_buf)  # fmt: skip
    ttnn.deallocate(x2)
    c = ttnn.slice(d, [0, 0, 0], [1, t_len, CH])
    u = ttnn.slice(d, [1, 0, 0], [2, t_len, CH])
    ttnn.deallocate(d)
    cc, uu = ttnn.multiply(c, 1.0 + cfm.inference_cfg_rate), ttnn.multiply(u, cfm.inference_cfg_rate)
    ttnn.deallocate(c)
    ttnn.deallocate(u)
    guided = ttnn.subtract(cc, uu)
    ttnn.deallocate(cc)
    ttnn.deallocate(uu)
    step = ttnn.multiply(guided, slot.dt_buf)
    ttnn.deallocate(guided)
    ttnn.add(slot.x_buf, step, output_tensor=slot.next_x)
    ttnn.deallocate(step)


def compile_slot(cfm, slot, t_len: int) -> None:
    """Every program the capture, `_reuse_trace` and the replay loop will use, compiled before any capture: a copy
    compiled while a trace is alive allocates a program-cache buffer the tracker flags (the note in `_capture`)."""
    for buf in (slot.mu2_buf, slot.spks2_buf, slot.cond2_buf, slot.mask2_buf, slot.x_buf):
        tmp = ttnn.clone(buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(tmp, buf)
        ttnn.deallocate(tmp)
    temb, dt = cfm._temb_device(0.0), cfm._dt_device(0.1)
    ttnn.copy(temb, slot.temb_raw_buf)
    ttnn.copy(dt, slot.dt_buf)
    ttnn.deallocate(temb)
    ttnn.deallocate(dt)
    for _ in range(2):
        body(cfm, slot, t_len)
    ttnn.copy(slot.next_x, slot.x_buf)
    z = ttnn.from_torch(cfm.rand_noise[:, :t_len, :].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=cfm.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    ttnn.copy(z, slot.x_buf)
    ttnn.deallocate(z)
    ttnn.synchronize_device(cfm.device)


def capture_slot(cfm, slot, t_len: int) -> None:
    slot.trace_id = ttnn.begin_trace_capture(cfm.device, cq_id=0)
    try:
        body(cfm, slot, t_len)
    finally:
        ttnn.end_trace_capture(cfm.device, slot.trace_id, cq_id=0)


def decode_graph_into(llm, out) -> None:
    """`TtQwen2LM._decode_graph` with the head's logits written into `out`, a buffer allocated before any capture."""
    x = llm.speech_embedding(llm._trace_tokens)
    x = ttnn.unsqueeze_to_4D(x)
    x = ttnn.to_memory_config(x, llm.args.get_residual_mem_config(Mode.DECODE, None))
    rot_mats = llm.rope_setup.get_rot_mats(llm._trace_rot_idxs)
    for layer in llm.layers:
        x = layer(x, llm._trace_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)
    x = llm.norm(x, mode=Mode.DECODE, norm_config=llm.args.get_norm_config("lm_head", Mode.DECODE, None))
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    head = llm.llm_decoder
    kw = {} if head.compute_config is None else {"compute_kernel_config": head.compute_config, "dtype": head.logits_dtype}
    ttnn.linear(x, head.weight, bias=head.bias, optional_output_tensor=out, **kw)
    ttnn.deallocate(x)


def prepare_llm_trace(llm, preallocate_output: bool):
    """`TtQwen2LM._decode_step_traced`'s first call, split: the persistent inputs and a compile step here, before
    any capture. Returns the pre-allocated logits buffer (or None)."""
    llm._trace_tokens = ttnn.from_torch(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=llm.mesh_device)  # fmt: skip
    llm._trace_pos = ttnn.from_torch(torch.tensor([0]), device=llm.mesh_device, dtype=ttnn.int32)
    llm._trace_rot_idxs = llm.rope_setup.get_rot_idxs(torch.tensor([0]), on_host=False)
    tokens_host = ttnn.from_torch(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
    ttnn.copy_host_to_device_tensor(tokens_host, llm._trace_tokens)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([0]), dtype=ttnn.int32), llm._trace_pos)
    ttnn.copy_host_to_device_tensor(llm.rope_setup.get_rot_idxs(torch.tensor([0]), on_host=True), llm._trace_rot_idxs)
    warm = llm._decode_graph()  # compiles the graph; its output doubles as the pre-allocated logits buffer
    if not preallocate_output:
        ttnn.deallocate(warm)
        warm = None
    else:
        decode_graph_into(llm, warm)  # compiles the linear that writes into it
    ttnn.synchronize_device(llm.mesh_device)
    return warm


def capture_llm_trace(llm, logits_buf) -> None:
    trace_id = ttnn.begin_trace_capture(llm.mesh_device, cq_id=0)
    if logits_buf is None:
        llm._trace_logits = llm._decode_graph()  # allocated inside the capture, as the pipeline does today
    else:
        decode_graph_into(llm, logits_buf)
        llm._trace_logits = logits_buf
    ttnn.end_trace_capture(llm.mesh_device, trace_id, cq_id=0)
    llm._trace_id = trace_id


def inference_streaming_freeing(self, token, prompt_token, prompt_feat, embedding, bucket_tokens, context_len=3):
    """`TtCausalMaskedDiffWithXvec.inference_streaming` with its device locals freed before the CFM runs: the token
    ids and embeddings after the encoder, the encoder's projection once it is on the host. Otherwise they are alive at
    the CFM trace's replay, and the tracker flags them (the prototype's first proof runs, with tracebacks)."""
    assert token.shape[0] == 1
    spks = self._xvec(embedding)
    full_token = torch.cat([prompt_token, token], dim=1)
    n_valid = full_token.shape[1] - context_len
    if bucket_tokens < full_token.shape[1]:
        raise ValueError(f"bucket_tokens {bucket_tokens} < {full_token.shape[1]} tokens (look-ahead included)")
    full_token = torch.nn.functional.pad(full_token, (0, bucket_tokens - full_token.shape[1]))
    ids_dev = ttnn.from_torch(full_token.reshape(1, 1, 1, -1).clamp(min=0).to(torch.int32), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)  # fmt: skip
    tok_emb_dev = self.input_embedding(ids_dev)
    h_dev = self.encoder(tok_emb_dev, bucket_tokens, 1, streaming=True, valid_length=n_valid, context_rows=context_len)
    ttnn.deallocate(ids_dev)
    ttnn.deallocate(tok_emb_dev)
    t_len2 = h_dev.shape[1]
    valid2 = n_valid * t_len2 // bucket_tokens
    mel_len1 = prompt_feat.shape[1]
    proj = ttnn.linear(h_dev, self.encoder_proj_w, bias=self.encoder_proj_b)
    ttnn.deallocate(h_dev)
    mu = ttnn.to_torch(proj).float().reshape(1, t_len2, self.output_size)
    ttnn.deallocate(proj)
    conds = torch.zeros(1, t_len2, self.output_size, dtype=mu.dtype)
    conds[:, :mel_len1] = prompt_feat
    mask = torch.zeros(1, t_len2, 1, dtype=mu.dtype)
    mask[:, :valid2] = 1.0
    feat = self.decoder.forward(mu, mask, self.n_timesteps, spks, conds, streaming=True)
    return feat[:, mel_len1:valid2, :]


def stream_six(pipe, ctxs, label: str) -> dict:
    out = {}
    for ctx in ctxs:
        case = ctx.meta["case"]
        rec = {"case": case["case_id"]}
        try:
            syn = pipe.synthesize_stream(ctx, case["text"], rng=RandomSources(llm_seed=1986, noise_seed=1))
            rec.update(ok=True, tokens=[int(t) for t in syn.tokens], audio=syn.audio.copy(), first_audio_s=syn.first_audio_s,
                       rtf=syn.rtf, audio_s=syn.audio_s, cfm_s=[c["cfm"] for c in syn.chunks],
                       flow_s=[c["flow"] for c in syn.chunks])  # fmt: skip
            print(f"  [{label}] {case['case_id']}: first audio {syn.first_audio_s:.3f} s, RTF {syn.rtf:.3f}, "
                  f"CFM {' + '.join(f'{c:.2f}' for c in rec['cfm_s'])}", flush=True)  # fmt: skip
        except Exception as e:  # noqa: BLE001 -- the tracker's verdict is what this run is for
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:20000]}")
            print(f"  [{label}] {case['case_id']}: FAILED {rec['error'][:400]}", flush=True)
        out[case["case_id"]] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=("cfm-first", "llm-first"), default="cfm-first")
    ap.add_argument("--llm-output", choices=("preallocated", "inside"), default="preallocated")
    ap.add_argument("--no-baseline", action="store_true", help="skip the eager baseline (a diagnostic run)")
    ap.add_argument("--limit", type=int, default=0, help="stream only the first N utterances (a diagnostic run)")
    ap.add_argument("--wav-dir", default=None, help="write each run's wavs here (eager/, traced/)")
    ap.add_argument("--record", action="store_true",
                    help="record every CFM call's input mu and output in both runs, and compare them call by call")
    ap.add_argument("--check-cfm", action="store_true",
                    help="after each traced solve, the eager solve of the same inputs, compared (a diagnostic run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tracking = ttnn.TRACE_ALLOC_TRACKING if hasattr(ttnn, "TRACE_ALLOC_TRACKING") else None
    ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(INPUTS, "*.npz")))]
    ctxs = [c for c in ctxs if c.meta["case"]["set"] == "librispeech"]
    if args.limit:
        ctxs = ctxs[: args.limit]
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=TRACE_REGION)
    report = {"order": args.order, "llm_output": args.llm_output, "tracking": bool(tracking),
              "trace_region_size": TRACE_REGION}  # fmt: skip
    try:
        pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
        t0 = time.perf_counter()
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        report["warmups_s"] = round(time.perf_counter() - t0, 1)
        rec = {"eager": [], "traced": []}
        label = ["eager"]
        base_forward = pipe.flow.decoder.forward  # the stage clock's wrapper

        def recording_forward(mu, mask, n_timesteps, spks, cond, use_trace=None, streaming=False):
            out = base_forward(mu, mask, n_timesteps, spks, cond, use_trace=use_trace, streaming=streaming)
            rec[label[0]].append((int(mu.shape[1]), bool(streaming), mu.clone(), out.clone()))
            return out

        if args.record:
            pipe.flow.decoder.forward = recording_forward
        eager = {} if args.no_baseline else stream_six(pipe, ctxs, "eager")
        label[0] = "traced"

        cfm, llm = pipe.flow.decoder, pipe.llm
        buckets = pipe.config.flow_token_buckets()
        report["dram_free_before_slots"] = dram_free(device)
        untraceable = raw_weight_convs(cfm.estimator, [2 * b for b in buckets])
        report["untraceable_buckets"] = {str(t): v for t, v in untraceable.items()}
        print(f"untraceable (a raw-weight conv verdict): {report['untraceable_buckets'] or 'none'}", flush=True)
        slots = {}
        for b in buckets:
            if 2 * b not in untraceable:
                slots[2 * b] = make_slot(cfm, 2 * b)
        logits_buf = prepare_llm_trace(llm, args.llm_output == "preallocated")
        report["dram_free_after_slots"] = dram_free(device)
        t0 = time.perf_counter()
        for t_len, slot in slots.items():
            compile_slot(cfm, slot, t_len)
        report["compile_bodies_s"] = round(time.perf_counter() - t0, 1)

        sizes = []
        def capture_cfms():
            for t_len, slot in slots.items():
                before = trace_bytes(device)
                t = time.perf_counter()
                capture_slot(cfm, slot, t_len)
                sizes.append({"trace": f"cfm T={t_len}", "bytes": trace_bytes(device) - before,
                              "capture_s": round(time.perf_counter() - t, 2)})  # fmt: skip
        def capture_llm():
            before = trace_bytes(device)
            capture_llm_trace(llm, logits_buf)
            sizes.append({"trace": "llm decode", "bytes": trace_bytes(device) - before})
        for step in ((capture_cfms, capture_llm) if args.order == "cfm-first" else (capture_llm, capture_cfms)):
            step()
        report["traces"] = sizes
        report["trace_region_used"] = trace_bytes(device)
        print(f"trace region: {report['trace_region_used'] / 1e6:.1f} MB of {TRACE_REGION / 1e6:.0f} MB for "
              f"{len(sizes)} traces", flush=True)  # fmt: skip

        # route the streamed chunks' CFM through the start-up traces; keep the decode trace across requests
        cfm._trace_capacity = len(slots) + 1
        cfm._traces.clear()
        for t_len, slot in slots.items():
            cfm._traces[cfm._trace_key_for(t_len, CH, True)] = slot
        timed_forward = cfm.forward  # the pipeline's stage clock wraps the instance attribute

        checks = report.setdefault("cfm_checks", [])

        def forward(mu, mask, n_timesteps, spks, cond, use_trace=None, streaming=False):
            traced = streaming and int(mu.shape[1]) in slots  # an untraceable bucket stays eager, never captures
            out = timed_forward(mu, mask, n_timesteps, spks, cond, use_trace=traced, streaming=streaming)
            if traced and args.check_cfm:
                ref = timed_forward(mu, mask, n_timesteps, spks, cond, use_trace=False, streaming=streaming)
                a, b = out.double().flatten().numpy(), ref.double().flatten().numpy()
                checks.append({"t_len": int(mu.shape[1]), "valid": int(mask.sum()), "pcc": float(np.corrcoef(a, b)[0, 1]),
                               "max_abs_diff": float(np.abs(a - b).max())})  # fmt: skip
                print(f"    cfm check T={mu.shape[1]} valid={int(mask.sum())}: PCC {checks[-1]['pcc']:.6f}", flush=True)
            return out

        cfm.forward = forward
        llm.release_decode_trace = lambda: None
        type(pipe.flow).inference_streaming = inference_streaming_freeing
        traced = stream_six(pipe, ctxs, "traced")

        rows = []
        for cid, t in traced.items():
            e = eager.get(cid)
            if e is None:
                rows.append({"case": cid, "traced_ok": t["ok"], **({} if t["ok"] else {"error": t["error"]})})
                print(json.dumps(rows[-1])[:3000], flush=True)
                continue
            row = {"case": cid, "traced_ok": t["ok"]}
            if t["ok"]:
                n = min(len(e["audio"]), len(t["audio"]))
                a, b = e["audio"][:n].astype(np.float64), t["audio"][:n].astype(np.float64)
                # Waveform PCC is not the measure here: the traced step blends and updates on the device in bf16,
                # the eager one on the host in fp32, and a 1e-4 mel difference drifts HiFT's sine phase. Log-mel L1
                # is the stage A gate's measure for exactly that (tests/e2e/test_streaming.py).
                from models.experimental.cosyvoice2.tests.pcc.test_hift_chunked import _logmel

                lm = float((_logmel(torch.from_numpy(t["audio"][:n].astype(np.float32))) -
                            _logmel(torch.from_numpy(e["audio"][:n].astype(np.float32)))).abs().mean())  # fmt: skip
                if args.wav_dir:
                    import soundfile

                    for side, audio in (("eager", e["audio"]), ("traced", t["audio"])):
                        os.makedirs(os.path.join(args.wav_dir, side), exist_ok=True)
                        soundfile.write(os.path.join(args.wav_dir, side, f"{cid}.wav"), audio, 24000)
                row.update(tokens_equal=e["tokens"] == t["tokens"], samples_equal=len(e["audio"]) == len(t["audio"]),
                           logmel_l1=round(lm, 4), waveform_pcc=round(float(np.corrcoef(a, b)[0, 1]), 4),
                           first_audio_s=[round(e["first_audio_s"], 4), round(t["first_audio_s"], 4)],
                           rtf=[round(e["rtf"], 4), round(t["rtf"], 4)],
                           cfm_s=[[round(x, 3) for x in e["cfm_s"]], [round(x, 3) for x in t["cfm_s"]]])  # fmt: skip
            else:
                row["error"] = t["error"]
            rows.append(row)
            print(json.dumps(row), flush=True)
        report["utterances"] = rows
        if args.record:
            calls = []
            for i, (e, t) in enumerate(zip(rec["eager"], rec["traced"])):
                def pcc(a, b):
                    return float(np.corrcoef(a.double().flatten().numpy(), b.double().flatten().numpy())[0, 1])
                calls.append({"call": i, "t_len": e[0], "streaming": e[1], "same_geometry": e[0] == t[0],
                              "mu_pcc": pcc(e[2], t[2]) if e[2].shape == t[2].shape else None,
                              "mu_equal": bool(torch.equal(e[2], t[2])) if e[2].shape == t[2].shape else False,
                              "out_pcc": pcc(e[3], t[3]) if e[3].shape == t[3].shape else None})  # fmt: skip
                print(json.dumps(calls[-1]), flush=True)
            report["calls"] = calls
    finally:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        if "pipe" in locals():  # release every trace this process captured, then the device
            if pipe.llm._trace_id is not None:
                ttnn.release_trace(device, pipe.llm._trace_id)
                pipe.llm._trace_id = None
            for slot in locals().get("slots", {}).values():
                if slot.trace_id is not None:
                    ttnn.release_trace(device, slot.trace_id)
                    slot.trace_id = None
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
