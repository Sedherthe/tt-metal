"""A fix candidate for B28, tested without changing the PR: streaming's final HiFT call padded IN FRONT (as the first
call already is) instead of at the end, so the call ends at the utterance's true end, as upstream's does, and the
silence sits under the crossfade with the previous chunk.

The cache source must then replace the source after the front padding. `TtHiFTGenerator.inference` replaces it at
index 0, so this script calls it twice for a final chunk: once for the model's own source over the padded frames,
then with [that source, the cache] as `cache_source`. The padded frames keep their own source and only the cache
region changes. The geometries are the end-padded ones (128 and 256 frames), so nothing new compiles.

Same cases, noise and outputs as you_tail_ab.py, as a third variant `front`:
- `front_1987/`: all six cases with R5's default noise, a run dir for scripts/eval_wer_sim.py;
- `you_sweep/front_seed*.wav`, `front_upnoise.wav`: 260-123440-0010 over the same noise seeds;
- `mech/front_*.wav`: upstream's mel, F0 and noise through the front-padded stream. Its tail and final seam against
  upstream's are in `summary_front.json`.

    python you_tail_front.py --out <the you_tail_ab.py dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile
import torch

import ttnn

from models.experimental.cosyvoice2.tt import streaming as st
from models.experimental.cosyvoice2.tt.hifigan.chunking import HOP, OVERLAP_FRAMES, fade_in_out
from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, CosyVoice2TTNN
from models.experimental.cosyvoice2.tt.prompt import PromptContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from you_tail_ab import INPUTS, REF, SEEDS, SR, YOU, dbfs, seeded, tail, upstream_noise  # noqa: E402


class FrontFinalHiFTStream(st.HiFTStream):
    """HiFTStream whose final call (when it has a cache) is padded in front."""

    def step(self, new_mel, final, noise, f0=None):
        if not final or self.mel_cache is None:
            return super().step(new_mel, final, noise, f0)
        mel = torch.cat([self.mel_cache, new_mel], dim=1)
        frames = int(mel.shape[1])
        run = next((b for b in (128, 256) if b >= frames), frames)
        front = run - frames
        mel_run = torch.cat([st.silence_mel(front), mel], dim=1)
        noise_run = torch.cat([torch.zeros(1, front * HOP, self.harmonics), noise], dim=1)
        f0_run = None if f0 is None else torch.cat([torch.zeros(1, front), f0.reshape(1, -1)], 1)
        mel_dev = ttnn.from_torch(mel_run, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.hift.device)
        wav0, own = self.hift.inference(mel_dev, run, 1, sine_noise=noise_run, f0=f0_run, return_source=True)
        ttnn.deallocate(wav0)
        cache = torch.cat([own[:, : front * HOP], self.source_cache.reshape(1, -1)], dim=1)
        wav_dev, _ = self.hift.inference(
            mel_dev, run, 1, sine_noise=noise_run, f0=f0_run, cache_source=cache, return_source=True
        )
        wav = ttnn.to_torch(wav_dev).float().reshape(-1)[front * HOP : (front + frames) * HOP]
        ttnn.deallocate(wav_dev)
        ttnn.deallocate(mel_dev)
        wav = fade_in_out(wav, self.speech_tail, self.window)
        self.calls.append({"frames": frames, "run": run, "front": front})
        self.mel_cache = self.source_cache = self.speech_tail = None
        return wav.numpy()


def pcc(a, b) -> float:
    return float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = sorted(p[: -len(".npz")] for p in os.listdir(REF) if p.startswith("zero_shot_") and p.endswith(".npz"))
    os.makedirs(f"{args.out}/front_1987", exist_ok=True)
    st.HiFTStream = FrontFinalHiFTStream  # StreamSession builds its HiFTStream from the module global
    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)
    summary = {"corpus": {}, "mech": {}, "you_sweep": {}}
    try:
        pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        dtype = getattr(ttnn, pipe.config.hift_source_dtype)
        rows = []
        for c in cases:
            ref = np.load(f"{REF}/{c}.npz")
            ctx = PromptContext.from_npz(f"{INPUTS}/{c}.npz")
            tokens, up = ref["tokens"].tolist(), ref["audio"]
            audio = np.concatenate([x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, seeded(1987, pipe.harmonics))])
            audio = audio.astype(np.float32)
            soundfile.write(f"{args.out}/front_1987/{c}.wav", audio, SR)
            rows.append({**json.loads(str(np.load(f"{INPUTS}/{c}.npz")["case_json"])), "wav": f"{c}.wav",
                         "audio_s": round(len(audio) / SR, 3), "segment_tokens": [tokens]})  # fmt: skip
            summary["corpus"][f"front/{c}"] = tail(audio, up)
            mech = FrontFinalHiFTStream(pipe.hift, pipe.harmonics, dtype=dtype)
            sched = st.stream_schedule(len(tokens), ctx.n_prompt_tokens)
            got = [mech.step(torch.from_numpy(ref[f"mel_{k}"]), ch.final, torch.from_numpy(ref[f"hift_noise_{k}"]),
                             f0=torch.from_numpy(ref[f"hift_f0_{k}"])) for k, ch in enumerate(sched)]  # fmt: skip
            m = np.concatenate(got).astype(np.float32)
            soundfile.write(f"{args.out}/mech/front_{c}.wav", m, SR)
            # the final seam: the last chunk's first 3,840 samples (the crossfade) and 480 either side, as the gate
            k = len(sched) - 1
            start = sum(len(ref[f"speech_{i}"]) for i in range(k))
            lo, hi = start - 480, start + OVERLAP_FRAMES * HOP + 480
            final_body = slice(start, len(up) - int(0.4 * SR))
            summary["mech"][f"front/{c}"] = {**tail(m, up), "final_seam_pcc": round(pcc(m[lo:hi], up[lo:hi]), 5),
                                             "final_chunk_pcc_before_tail": round(pcc(m[final_body], up[final_body]), 5)
                                             if final_body.stop - final_body.start > 960 else None}  # fmt: skip
            print(f"front {c}: ours {summary['corpus'][f'front/{c}']}; mech {summary['mech'][f'front/{c}']}", flush=True)
            if c == YOU:
                for s in SEEDS[1:]:
                    a = np.concatenate([x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, seeded(s, pipe.harmonics))])
                    soundfile.write(f"{args.out}/you_sweep/front_seed{s}.wav", a.astype(np.float32), SR)
                    summary["you_sweep"][f"front/seed{s}"] = tail(a, up)
                a = np.concatenate([x.audio for x in st.stream_fixed_tokens(pipe, ctx, tokens, upstream_noise(ref))])
                soundfile.write(f"{args.out}/you_sweep/front_upnoise.wav", a.astype(np.float32), SR)
                summary["you_sweep"]["front/upnoise"] = tail(a, up)
        with open(f"{args.out}/front_1987/results.json", "w") as fh:
            json.dump({"backend": "ttnn-streaming-offline-front-final", "results": rows}, fh, indent=2)
    finally:
        pipe.release() if "pipe" in locals() else None
        ttnn.close_device(device)
    with open(f"{args.out}/summary_front.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
