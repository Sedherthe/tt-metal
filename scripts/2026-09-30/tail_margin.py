"""D41's "last 0.4 s at least 20 dB below the signal" criterion: its measured distribution, and what limits it.

Stage A's mechanism (upstream's mel, F0 and noise per HiFT call through `HiFTStream`) over six references: the
suite's (`cosyvoice2_streaming_ref`, noise seed 1986) and D43's five noise draws (`cosyvoice2_draws/ref_stream_seed1-5`),
six cases each, so 36 final chunks. Three variants (`--variants`):
- `masked`: the final call as built (masked end padding);
- `silence`: the final call padded with silence and not masked (the code before `ed1c3ad1c5`), rebuilt here by a
  subclass, to check a threshold still separates the two;
- `exact`: the final call unpadded, at its real length (`FINAL_CALL_BUCKETS = ()`), compiled for each length: the
  port's own error with no padding at all, the floor the masked call is measured against.

Per final chunk: the last 20 ms against upstream's; over the last 0.4 s the signal's and the difference's RMS
(dBFS) and their margin; the same margin over the chunk before those 0.4 s (the port's own error, for scale); and
the last 0.4 s in 20 ms frames (signal and difference), to see which part limits the margin.

    python tail_margin.py --out <json> [--variants masked,silence,exact]
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
from models.experimental.cosyvoice2.tt.hifigan.chunking import HOP, fade_in_out
from models.experimental.cosyvoice2.tt.prompt import PromptContext

INPUTS = "/home/user/data/cosyvoice2_inputs"
REFS = ["/home/user/data/cosyvoice2_streaming_ref"] + [f"/home/user/data/cosyvoice2_draws/ref_stream_seed{s}" for s in range(1, 6)]
SR, N20, N400 = 24000, 480, 9600


def dbfs(x) -> float:
    x = np.asarray(x, np.float64)
    return float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12)))


class SilencePaddedFinal(st.HiFTStream):
    """The final call as it was before `ed1c3ad1c5`: padded at its end with silence mel, no masks, no end gain."""

    def step(self, new_mel, final, noise, f0=None):
        if not final:
            return super().step(new_mel, final, noise, f0)
        first = self.mel_cache is None
        mel = new_mel if first else torch.cat([self.mel_cache, new_mel], dim=1)
        frames = int(mel.shape[1])
        run = next((b for b in st.FINAL_CALL_BUCKETS if b >= frames), frames)
        back = run - frames
        mel_run = torch.cat([mel, st.silence_mel(back)], dim=1)
        noise_run = torch.cat([noise, torch.zeros(1, back * HOP, self.harmonics)], dim=1)
        f0_run = None if f0 is None else torch.cat([f0.reshape(1, -1), torch.zeros(1, back)], 1)
        mel_dev = ttnn.from_torch(mel_run, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.hift.device)
        wav_dev, _ = self.hift.inference(mel_dev, run, 1, sine_noise=noise_run, f0=f0_run,
                                         cache_source=self.source_cache, return_source=True)  # fmt: skip
        wav = ttnn.to_torch(wav_dev).float().reshape(-1)[: frames * HOP]
        ttnn.deallocate(wav_dev)
        ttnn.deallocate(mel_dev)
        if self.speech_tail is not None:
            wav = fade_in_out(wav, self.speech_tail, self.window)
        self.mel_cache = self.source_cache = self.speech_tail = None
        return wav.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="masked,silence")
    args = ap.parse_args()
    variants = {"masked": st.HiFTStream, "silence": SilencePaddedFinal, "exact": st.HiFTStream}
    chosen = [(v, variants[v]) for v in args.variants.split(",")]
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
    from models.experimental.cosyvoice2.tt.hifigan.conv import config_tensors_in_dram_override
    from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.experimental.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    rows = []
    try:
        with config_tensors_in_dram_override(True):  # as the pipeline builds it
            sd = load_checkpoint_file("hift.pt")
            decode_ref = TorchHiFTDecodeRef.from_checkpoint(sd)
            ref = TorchHiFTGeneratorInferenceRef(decode_ref, TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(sd, "f0_predictor.")),
                                                 sd["m_source.l_linear.weight"], sd["m_source.l_linear.bias"])  # fmt: skip
            gen = TtHiFTGenerator(device, ref, TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32), dtype=ttnn.float32)
        harmonics = ref.harmonic_num + 1
        for ref_dir in REFS:
            for path in sorted(p for p in os.listdir(ref_dir) if p.startswith("zero_shot_") and p.endswith(".npz")):
                case = path[: -len(".npz")]
                d = np.load(os.path.join(ref_dir, path))
                ctx = PromptContext.from_npz(os.path.join(INPUTS, path))
                sched = st.stream_schedule(len(d["tokens"]), ctx.n_prompt_tokens)
                k_final = len(sched) - 1
                want = d[f"speech_{k_final}"].astype(np.float64)
                for variant, cls in chosen:
                    st.FINAL_CALL_BUCKETS = () if variant == "exact" else (128, 256)
                    stream = cls(gen, harmonics, dtype=ttnn.float32)
                    got = None
                    for k, ch in enumerate(sched):
                        out = stream.step(torch.from_numpy(d[f"mel_{k}"]), ch.final, torch.from_numpy(d[f"hift_noise_{k}"]),
                                          f0=torch.from_numpy(d[f"hift_f0_{k}"]))  # fmt: skip
                        if k == k_final:
                            got = out.astype(np.float64)
                    n4 = min(N400, len(want))
                    tail_sig, tail_diff = dbfs(want[-n4:]), dbfs(got[-n4:] - want[-n4:])
                    body = slice(0, len(want) - n4)
                    frames = [(round(dbfs(want[len(want) - n4 + i: len(want) - n4 + i + N20]), 1),
                               round(dbfs(got[len(want) - n4 + i: len(want) - n4 + i + N20] - want[len(want) - n4 + i: len(want) - n4 + i + N20]), 1))
                              for i in range(0, n4, N20)]  # fmt: skip
                    row = {
                        "ref": os.path.basename(ref_dir), "case": case, "variant": variant, "final_chunk_s": round(len(want) / SR, 2),
                        "last20_db": round(dbfs(got[-N20:]), 1), "last20_up_db": round(dbfs(want[-N20:]), 1),
                        "tail_signal_db": round(tail_sig, 1), "tail_diff_db": round(tail_diff, 1), "tail_margin_db": round(tail_sig - tail_diff, 1),
                        "body_signal_db": round(dbfs(want[body]), 1) if body.stop > N20 else None,
                        "body_margin_db": round(dbfs(want[body]) - dbfs(got[body] - want[body]), 1) if body.stop > N20 else None,
                        "tail_frames": frames,
                    }  # fmt: skip
                    rows.append(row)
                    print(f"{row['ref']:24s} {case:28s} {variant:8s} last 20 ms {row['last20_db']:7.1f} / {row['last20_up_db']:7.1f}; last 0.4 s "
                          f"signal {tail_sig:6.1f} diff {tail_diff:6.1f} margin {row['tail_margin_db']:5.1f}; body margin {row['body_margin_db']}", flush=True)  # fmt: skip
    finally:
        ttnn.close_device(device)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    for variant, _ in chosen:
        m = sorted(r["tail_margin_db"] for r in rows if r["variant"] == variant)
        e = sorted(abs(r["last20_db"] - r["last20_up_db"]) for r in rows if r["variant"] == variant)
        print(f"{variant}: last 0.4 s margin over {len(m)} final chunks: min {m[0]}, 5 lowest {m[:5]}, median {m[len(m) // 2]}, max {m[-1]}; "
              f"last 20 ms |dB diff|: max {e[-1]:.1f}, median {e[len(e) // 2]:.1f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
