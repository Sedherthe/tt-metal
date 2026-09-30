"""Score every wav in a directory with the PR's scorer call (you_asr.py's `run`: the scorer's exact transcribe, with the
first text token's decision recorded), and report which end in "you". For you_tail_ab.py's `you_sweep/` (and
`mech/`), in the reference venv.

    python you_score_sweep.py <dir> [<dir> ...] --out <jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import you_asr  # noqa: E402  (imports the PR's scorer, unmodified)

ews = you_asr.ews


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer

    record = []
    orig_update = whisper.decoding.GreedyDecoder.update

    def update(self, tokens, logits, sum_logprobs):
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top = torch.topk(logprobs[0], 4)
        chosen = int(logits[0].argmax())
        record.append((chosen, float(logprobs[0, chosen]), list(zip(top.indices.tolist(), top.values.tolist()))))
        return orig_update(self, tokens, logits, sum_logprobs)

    whisper.decoding.GreedyDecoder.update = update
    asr = ews.ASR()
    tok = get_tokenizer(asr.model.is_multilingual, num_languages=asr.model.num_languages, language="en",
                        task="transcribe")  # fmt: skip
    with open(args.out, "w") as fh:
        for d in args.dirs:
            for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
                r = you_asr.run(asr, ews.load_16k_mono(p).numpy(), tok, record)
                first = next(s for s in r["steps"] if not s["token"].startswith("<|"))
                you = r["text"].rstrip(" .").lower().endswith(" you")
                margin = first["logprob"] - first["runners_up"][0][1]
                fh.write(json.dumps({"wav": p, "you": you, "first": first["token"], "margin": round(margin, 3), **r}) + "\n")
                print(f"{os.path.basename(d)}/{os.path.basename(p):40s} you={'YES' if you else 'no ':3s} "
                      f"first={first['token']!r} margin {margin:.3f}  ...{r['text'][-34:]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
