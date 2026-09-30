"""Stage 3, step 3: the Euler step sweep summarized.

- Quality: corpus WER and SIM, mean (range) over the five noise draws, per step count, for TT and the reference,
  Stage 1 and streaming (`eval_draws.py`'s JSON). Then every utterance whose WER differs from its 10-step value in
  any draw, with the transcripts.
- Latency, TT only, from `steps_draws.py`'s records (one process, both warm-ups, nothing else on the host):
  - streaming: first audio and RTF (worst utterance, and the aggregate per draw), the first chunk's flow and CFM;
  - Stage 1: RTF, worst and aggregate.

    python3 steps_summary.py --scores <steps.json> --tt <steps dir> > steps_summary.md
"""
import argparse
import json
import os

import numpy as np

STEPS = (10, 8, 6, 5)
D43 = "/home/user/data/cosyvoice2_draws"


def spread(x, f="{:.3f}"):
    return f"{f.format(min(x))}–{f.format(max(x))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="eval_draws.py's JSON; without it, latency only")
    ap.add_argument("--tt", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.scores:
        tokens_check(args.tt, seeds)
        quality(json.load(open(args.scores)))
    latency(args.tt, seeds)


def tokens_check(tt, seeds):
    """Only the step count may differ from D43: every run's tokens against D43's for the same side and mode."""

    def tokens(path):
        return {r["case_id"]: r["segment_tokens"] for r in json.load(open(path))["results"]}

    same, runs = 0, 0
    for side in ("tt", "ref"):
        for mode in ("stage1", "stream"):
            want = tokens(os.path.join(D43, f"{side}_{mode}_seed1", "results.json"))
            for k in STEPS[1:]:
                for seed in seeds:
                    runs += 1
                    same += tokens(os.path.join(tt, f"steps{k}", f"{side}_{mode}_seed{seed}", "results.json")) == want
    print(f"Tokens: {same} of {runs} runs at 8, 6 and 5 steps sampled D43's tokens for their side and mode.\n")


def quality(sc):

    def cell(group, key):
        s = sc[group]["corpus"][key]
        return f"{s['mean']:.2f} ({s['min']:.2f}–{s['max']:.2f})"

    print("## Quality: corpus WER % and SIM, mean (range) over five noise draws\n")
    print("| Euler steps | WER, TT Stage 1 | WER, reference Stage 1 | WER, TT streaming | WER, reference streaming | "
          "SIM, TT Stage 1 | SIM, reference Stage 1 | SIM, TT streaming | SIM, reference streaming |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|")
    for k in STEPS:
        groups = [f"TT stage1 {k}", f"reference stage1 {k}", f"TT stream {k}", f"reference stream {k}"]
        print(f"| {k} | " + " | ".join(cell(g, "wer_percent") for g in groups) + " | "
              + " | ".join(cell(g, "sim") for g in groups) + " |")  # fmt: skip

    print("\n## Utterances whose WER differs from 10 steps in any draw\n")
    moved = False
    for side in ("TT", "reference"):
        for mode in ("stage1", "stream"):
            base = sc[f"{side} {mode} 10"]["cases"]
            for k in STEPS[1:]:
                for case, c in sc[f"{side} {mode} {k}"]["cases"].items():
                    b = base[case]
                    if c["wer_percent"] != b["wer_percent"]:
                        moved = True
                        print(f"- **{side} {mode}, {k} steps, {case}** ({c['words']} words): WER {c['wer_percent']} "
                              f"against {b['wer_percent']} at 10")  # fmt: skip
                        for i, (h, h10) in enumerate(zip(c["hypotheses"], b["hypotheses"])):
                            if h != h10:
                                print(f"  - draw {i + 1}: \"{h}\"\n    at 10: \"{h10}\"")
    if not moved:
        print("None: every utterance's WER is its 10-step value in every draw, on both sides and in both modes.")


def latency(tt, seeds):
    args = argparse.Namespace(tt=tt)
    print("\n## Latency (TT; `steps_draws.py`, one process, both warm-ups, nothing else on the host)\n")
    print("| Euler steps | first audio s, all 30 | first audio, worst per draw | streaming RTF, worst per draw | "
          "streaming RTF, aggregate per draw | first chunk: flow s | first chunk: CFM s | Stage 1 RTF, worst per draw | "
          "Stage 1 RTF, aggregate per draw |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|")
    for k in STEPS:
        first, worst_first, worst_rtf, agg_rtf, flow0, cfm0, s1_worst, s1_agg = [], [], [], [], [], [], [], []
        for seed in seeds:
            stream = json.load(open(os.path.join(args.tt, f"steps{k}", f"tt_stream_seed{seed}", "results.json")))["results"]
            stage1 = json.load(open(os.path.join(args.tt, f"steps{k}", f"tt_stage1_seed{seed}", "results.json")))["results"]
            f = [r["first_audio_s"] for r in stream]
            first += f
            worst_first.append(max(f))
            worst_rtf.append(max(r["rtf"] for r in stream))
            agg_rtf.append(sum(r["wall_s"] for r in stream) / sum(r["audio_s"] for r in stream))
            flow0 += [r["chunks"][0]["flow"] for r in stream]
            cfm0 += [r["chunks"][0]["cfm"] for r in stream]
            s1_worst.append(max(r["rtf"] for r in stage1))
            s1_agg.append(sum(r["wall_s"] for r in stage1) / sum(r["audio_s"] for r in stage1))
        print(f"| {k} | {spread(first)} | {spread(worst_first)} | {spread(worst_rtf)} | {spread(agg_rtf)} | "
              f"{spread(flow0, '{:.2f}')} | {spread(cfm0, '{:.2f}')} | {spread(s1_worst)} | {spread(s1_agg)} |")  # fmt: skip

    print("\n## Per utterance, streaming (TT): first audio and RTF, min–max over the five draws\n")
    cases = [r["case_id"] for r in json.load(open(os.path.join(args.tt, "steps10", "tt_stream_seed1", "results.json")))["results"]]
    print("| utterance | " + " | ".join(f"{k} steps: first audio / RTF" for k in STEPS) + " |")
    print("|---|" + "---|" * len(STEPS))
    for case in cases:
        cells = []
        for k in STEPS:
            rows = [next(r for r in json.load(open(os.path.join(args.tt, f"steps{k}", f"tt_stream_seed{s}", "results.json")))["results"]
                         if r["case_id"] == case) for s in seeds]  # fmt: skip
            cells.append(f"{spread([r['first_audio_s'] for r in rows])} / {spread([r['rtf'] for r in rows])}")
        print(f"| {case[len('zero_shot_'):]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
