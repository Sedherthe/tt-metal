"""The "you" clip: does Whisper's trailing "you" on 260-123440-0010 come and go, and how close is the decision?

Runs in the reference venv. It uses the PR's scorer (scripts/eval_wer_sim.py) as it is: its audio loader and its
exact `transcribe` call (large-v3, CPU, fp16 off, temperature 0.0 only, greedy, language given). On top of that it
records, at every greedy step, the chosen token's log-probability and the runner-up's, after Whisper's own logit
filters. That is the decision margin.

    python you_asr.py --mode normal|seeded --tag <run tag> [--splices] --out <jsonl>

`seeded`: torch, numpy and random seeded, deterministic algorithms on; the decode options are the scorer's (it
already disables temperature fallback: a single float temperature gives one pass, no fallback).
`--splices`: also transcribes our streamed clip with its last T seconds taken from upstream's, and the reverse.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "/home/user/tt-metal/models/experimental/cosyvoice2/scripts")
import eval_wer_sim as ews  # noqa: E402  (the PR's scorer, unmodified)

CASE = "zero_shot_260-123440-0010"
D = "/home/user/data"
CLIPS = {
    "ours_live": f"{D}/cosyvoice2_runs/0929/r5_stream_warm1/{CASE}.wav",  # R5; warm2 is bit-identical
    "ours_offline": f"{D}/cosyvoice2_runs/0929/stream_offline/{CASE}.wav",  # R3: our flow and HiFT, upstream's noise
    "upstream_stream": f"{D}/cosyvoice2_streaming_ref/{CASE}.wav",  # upstream streaming, the same tokens
    "ours_stage1": f"{D}/cosyvoice2_runs/0929/stage1_r1/{CASE}.wav",  # TT non-streaming, the same tokens
}
SPLICE_T = (0.1, 0.2, 0.4, 0.8)


def run(asr, audio16, tokenizer, record):
    """The scorer's exact call, with the greedy decoder's per-step choices recorded."""
    record.clear()
    out = asr.model.transcribe(audio16, language="en", fp16=False, temperature=0.0, beam_size=None)
    segs = [
        {k: (round(s[k], 4) if isinstance(s[k], float) else s[k])
         for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio", "temperature")}
        for s in out["segments"]
    ]  # fmt: skip
    steps = []
    for chosen, lp, top in record:
        name = lambda t: tokenizer.decode_with_timestamps([t]) if t != tokenizer.eot else "<|endoftext|>"  # noqa: E731
        alt = [(name(t), round(v, 4)) for t, v in top if t != chosen][:3]
        steps.append({"token": name(chosen), "logprob": round(lp, 4), "runners_up": alt})
    return {"text": out["text"].strip(), "segments": segs, "steps": steps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("normal", "seeded"), default="normal")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splices", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torchaudio
    import whisper
    from whisper.tokenizer import get_tokenizer

    if args.mode == "seeded":
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.use_deterministic_algorithms(True)

    record = []
    orig_update = whisper.decoding.GreedyDecoder.update

    def update(self, tokens, logits, sum_logprobs):  # logits are after Whisper's logit filters
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top = torch.topk(logprobs[0], 4)
        chosen = int(logits[0].argmax())
        record.append((chosen, float(logprobs[0, chosen]), list(zip(top.indices.tolist(), top.values.tolist()))))
        return orig_update(self, tokens, logits, sum_logprobs)

    whisper.decoding.GreedyDecoder.update = update
    asr = ews.ASR()
    tok = get_tokenizer(asr.model.is_multilingual, num_languages=asr.model.num_languages, language="en",
                        task="transcribe")  # fmt: skip

    rows = []
    for name, path in CLIPS.items():
        r = run(asr, ews.load_16k_mono(path).numpy(), tok, record)
        rows.append({"tag": args.tag, "mode": args.mode, "threads": torch.get_num_threads(), "clip": name, **r})
        print(f"[{args.tag} {args.mode}] {name:16s} {r['text']!r}", flush=True)

    if args.splices:
        import soundfile

        ours, sr = soundfile.read(CLIPS["ours_live"], dtype="float32")
        up, _ = soundfile.read(CLIPS["upstream_stream"], dtype="float32")
        assert len(ours) == len(up)
        for t in SPLICE_T:
            n = int(round(t * sr))
            for name, a in ((f"ours+upstream_last_{t}s", np.concatenate([ours[:-n], up[-n:]])),
                            (f"upstream+ours_last_{t}s", np.concatenate([up[:-n], ours[-n:]]))):  # fmt: skip
                a16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
                r = run(asr, a16, tok, record)
                rows.append({"tag": args.tag, "mode": args.mode, "threads": torch.get_num_threads(), "clip": name, **r})
                print(f"[{args.tag} {args.mode}] {name:26s} {r['text']!r}", flush=True)

    with open(args.out, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
