"""Summarize you_asr.py's JSONL: per run and clip, whether the transcript ends in "you", the first text token's
decision (" how" or " How", and its margin over the other), and the trailing " you" token's log-probability.

    python3 you_asr_summary.py <you_asr.jsonl>
"""
import collections
import json
import sys


def first_text_step(steps):
    for s in steps:
        if not s["token"].startswith("<|"):
            return s
    return None


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    print("| run | mode | threads | clip | ends in \"you\" | first token (logprob) | runner-up (logprob) | margin | \" you\" logprob |")
    print("|---|---|---|---|---|---|---|---|---|")
    tally = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        you = r["text"].rstrip(" .").lower().endswith(" you")
        f = first_text_step(r["steps"])
        alt = f["runners_up"][0] if f else ["", float("nan")]
        you_steps = [s for s in r["steps"] if s["token"] == " you"]
        you_lp = f"{you_steps[-1]['logprob']:.3f}" if you_steps else ""
        print(f"| {r['tag']} | {r['mode']} | {r['threads']} | {r['clip']} | {'yes' if you else 'no'} | "
              f"{f['token']!r} ({f['logprob']:.3f}) | {alt[0]!r} ({alt[1]:.3f}) | {f['logprob'] - alt[1]:.3f} | {you_lp} |")
        tally[r["clip"]][0] += you
        tally[r["clip"]][1] += 1
    print("\n| clip | runs ending in \"you\" |\n|---|---|")
    for clip, (y, n) in tally.items():
        print(f"| {clip} | {y} of {n} |")


if __name__ == "__main__":
    main()
