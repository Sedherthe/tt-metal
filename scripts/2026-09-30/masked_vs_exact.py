"""B28's fix against its reference, on the device: HiFT padded to its bucket and masked (tt/hifigan/valid_length.py)
against the same HiFT at the exact length, for every streamed final call and Stage 1's padded utterances.

Two arms, because the F0 predictor's convs round differently at another length and SineGen2 integrates F0 into the
sine phase over the whole call (the host proof, tests/pcc/test_hift_masked.py, shows the same on torch):
- same F0: one F0 injected on both sides (streaming: upstream's, the stage A mechanism; Stage 1: torch's F0
  predictor on the mel), so only the masking differs;
- own F0: each side's own F0 predictor, as the pipeline runs.

Streaming: stage A (`stream_fixed_tokens`) on TT's Stage 1 tokens with R5's noise, masked (as built) against
`FINAL_CALL_BUCKETS = ()` (the final call at its exact length); the mechanism arm feeds `HiFTStream` upstream's mel,
F0 and noise. Stage 1: TT's non-streaming mel of the same tokens, `inference_padded` against `inference` at the exact
length, one noise draw. The table: max |diff| and PCC over the whole utterance and over its last 20 ms.

    python masked_vs_exact.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import ttnn

from models.experimental.cosyvoice2.tt import streaming as st
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN, bucket_at_least
from models.experimental.cosyvoice2.tt.prompt import PromptContext

INPUTS = "/home/user/data/cosyvoice2_inputs"
REF = "/home/user/data/cosyvoice2_streaming_ref"
N20 = 480


def stats(got: np.ndarray, want: np.ndarray) -> dict:
    got, want = got.astype(np.float64), want.astype(np.float64)
    out = {}
    for name, sl in (("whole", slice(None)), ("last_20ms", slice(-N20, None))):
        g, w = got[sl], want[sl]
        out[name] = {"max_diff": float(np.abs(g - w).max()), "pcc": float(np.corrcoef(g, w)[0, 1])}
    return out


def seeded(seed: int, harmonics: int):
    gen = torch.Generator().manual_seed(seed)
    return lambda k, samples: torch.randn(1, samples, harmonics, generator=gen)


def mechanism(pipe, ref, sched, dtype) -> np.ndarray:
    stream = st.HiFTStream(pipe.hift, pipe.harmonics, dtype=dtype)
    return np.concatenate([stream.step(torch.from_numpy(ref[f"mel_{k}"]), ch.final, torch.from_numpy(ref[f"hift_noise_{k}"]),
                                       f0=torch.from_numpy(ref[f"hift_f0_{k}"])) for k, ch in enumerate(sched)])  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = sorted(p[: -len(".npz")] for p in os.listdir(REF) if p.startswith("zero_shot_") and p.endswith(".npz"))
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)  # the demo's
    rows = []
    try:
        from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
        from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef

        f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(load_checkpoint_file("hift.pt"), "f0_predictor."))
        pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        dtype = getattr(ttnn, pipe.config.hift_source_dtype)
        buckets = pipe.config.hift_frame_buckets()
        for c in cases:
            ref = np.load(f"{REF}/{c}.npz")
            ctx = PromptContext.from_npz(f"{INPUTS}/{c}.npz")
            tokens = ref["tokens"].tolist()
            sched = st.stream_schedule(len(tokens), ctx.n_prompt_tokens)
            final_frames = 2 * sched[-1].hop + (8 if len(sched) > 1 else 0)
            out = {}
            for variant, bk in (("masked", (128, 256)), ("exact", ())):
                st.FINAL_CALL_BUCKETS = bk
                out[f"mech_{variant}"] = mechanism(pipe, ref, sched, dtype)
                out[f"own_{variant}"] = np.concatenate(
                    [x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, seeded(1987, pipe.harmonics))]
                )
            st.FINAL_CALL_BUCKETS = (128, 256)
            # the final chunk against upstream's (the stage A gate's chunk PCC, tail included), masked and exact
            start = sum(len(ref[f"speech_{k}"]) for k in range(len(sched) - 1))
            want = ref[f"speech_{len(sched) - 1}"].astype(np.float64)
            final_pcc = {v: float(np.corrcoef(out[f"mech_{v}"][start:].astype(np.float64), want)[0, 1]) for v in ("masked", "exact")}
            rows.append({"case": c, "path": "streaming final chunk vs upstream (mechanism)", "arm": "PCC, whole chunk", "final_chunk_pcc": final_pcc})
            print(json.dumps(rows[-1]), flush=True)
            for arm in ("mech", "own"):
                s = stats(out[f"{arm}_masked"], out[f"{arm}_exact"])
                rows.append({"case": c, "path": f"streaming final call ({final_frames} frames, bucket "
                             f"{next(b for b in (128, 256) if b >= final_frames)})", "arm": "same F0" if arm == "mech" else "own F0", **s})  # fmt: skip
                print(json.dumps(rows[-1]), flush=True)
            # Stage 1, the padded single call
            mel = pipe.tokens_to_mel(tokens, ctx)
            frames = int(mel.shape[1])
            if frames >= buckets[-1]:
                continue
            run = bucket_at_least(frames, buckets)
            noise = torch.randn(1, frames * pipe.hift.upsample_scale, pipe.harmonics, generator=torch.Generator().manual_seed(frames))
            with torch.no_grad():
                f0 = f0_ref(mel.transpose(1, 2))
            for arm, f0_arm in (("same F0", f0), ("own F0", None)):
                masked = pipe.hift.inference_padded(mel, noise, run, f0=f0_arm).numpy()
                mel_dev = ttnn.from_torch(mel, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
                wav_dev = pipe.hift.inference(mel_dev, frames, 1, sine_noise=noise, f0=f0_arm)
                exact = ttnn.to_torch(wav_dev).float().reshape(-1)[: frames * pipe.hift.upsample_scale].numpy()
                ttnn.deallocate(wav_dev)
                ttnn.deallocate(mel_dev)
                rows.append({"case": c, "path": f"Stage 1 single call ({frames} frames, bucket {run})", "arm": arm, **stats(masked, exact)})
                print(json.dumps(rows[-1]), flush=True)
    finally:
        pipe.release() if "pipe" in locals() else None
        ttnn.close_device(device)
    with open(f"{args.out}/masked_vs_exact.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n| case | path | arm | whole: max\\|diff\\| | whole: PCC | last 20 ms: max\\|diff\\| | last 20 ms: PCC |\n|---|---|---|---|---|---|---|")
    for r in rows:
        if "final_chunk_pcc" in r:
            continue
        w, e = r["whole"], r["last_20ms"]
        print(f"| {r['case'][10:]} | {r['path']} | {r['arm']} | {w['max_diff']:.2e} | {w['pcc']:.6f} | {e['max_diff']:.2e} | {e['pcc']:.6f} |")
    print("\n| case | final chunk vs upstream, whole chunk PCC: masked | exact |\n|---|---|---|")
    for r in rows:
        if "final_chunk_pcc" in r:
            print(f"| {r['case'][10:]} | {r['final_chunk_pcc']['masked']:.5f} | {r['final_chunk_pcc']['exact']:.5f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
