"""The figures behind B34, from `tail_margin.py`'s two outputs (`tail_margin.json`: masked and silence;
`tail_margin_exact.json`: the final call at its exact length), 36 final chunks per variant.

1. The last 0.4 s's margin per variant (all 36), and per utterance: masked, exact, and their difference.
2. 121-127105-0015: the loudest 20 ms frame of the window, its share of the signal's and the difference's energy,
   and its own margin against the window's.
3. How far back the silence padding reaches: the earliest 20 ms frame whose difference exceeds the masked one's by
   more than 3 dB.
4. The same margin over the last 0.2 s and 0.1 s (from the 20 ms frames), masked and silence.

    python3 tail_margin_analysis.py > tail_margin_analysis.md
"""
import json
import math
import statistics
from collections import defaultdict

rows = json.load(open("tail_margin.json")) + json.load(open("tail_margin_exact.json"))
by = defaultdict(dict)
for r in rows:
    by[(r["case"][len("zero_shot_"):], r["ref"])][r["variant"]] = r


def pw(db):
    return 10 ** (db / 10)


def window_margin(frames, n):
    fr = frames[-n:]
    return 10 * math.log10(sum(pw(s) for s, _ in fr) / sum(pw(d) for _, d in fr))


print("## 1. The last 0.4 s: difference below the signal, dB\n")
for v in ("masked", "exact", "silence"):
    m = sorted(r["tail_margin_db"] for r in rows if r["variant"] == v)
    print(f"- {v}: {m[0]}–{m[-1]}, median {m[len(m) // 2]}, five lowest {m[:5]}")
print("\n| utterance | masked: range, mean ± sd | exact: range, mean ± sd | masked − exact | silence |")
print("|---|---|---|---|---|")
for case in sorted({c for c, _ in by}):
    refs = sorted(r for c, r in by if c == case)
    ms = [by[(case, r)]["masked"]["tail_margin_db"] for r in refs]
    es = [by[(case, r)]["exact"]["tail_margin_db"] for r in refs]
    ss = [by[(case, r)]["silence"]["tail_margin_db"] for r in refs]
    d = [a - b for a, b in zip(ms, es)]
    print(f"| {case} | {min(ms)}–{max(ms)}, {statistics.mean(ms):.1f} ± {statistics.stdev(ms):.1f} | "
          f"{min(es)}–{max(es)}, {statistics.mean(es):.1f} ± {statistics.stdev(es):.1f} | {min(d):+.1f} to {max(d):+.1f} | "
          f"{min(ss)}–{max(ss)} |")  # fmt: skip

print("\n## 2. 121-127105-0015: the loudest 20 ms frame of the window\n")
print("| reference | variant | window margin | frame (s before the end) | frame signal dBFS | share of signal energy | "
      "share of difference energy | frame margin |")  # fmt: skip
print("|---|---|---|---|---|---|---|---|")
for (case, ref), v in sorted(by.items()):
    if case != "121-127105-0015":
        continue
    for variant in ("masked", "exact"):
        fr = v[variant]["tail_frames"]
        ps, pd = [pw(s) for s, _ in fr], [pw(d) for _, d in fr]
        i = max(range(len(fr)), key=lambda k: ps[k])
        print(f"| {ref} | {variant} | {v[variant]['tail_margin_db']} | {0.4 - 0.02 * i:.2f}–{0.38 - 0.02 * i:.2f} | "
              f"{fr[i][0]:.0f} | {100 * ps[i] / sum(ps):.0f} % | {100 * pd[i] / sum(pd):.0f} % | {fr[i][0] - fr[i][1]:.0f} |")  # fmt: skip

print("\n## 3. How far back the silence padding reaches\n")
reach = defaultdict(set)
for (case, ref), v in by.items():
    m, s = v["masked"]["tail_frames"], v["silence"]["tail_frames"]
    n = len(m)
    reach[case].add(next((20 * (n - i) for i in range(n) if s[i][1] > m[i][1] + 3), 0))
for case, ms in sorted(reach.items()):
    print(f"- {case}: its difference first exceeds the masked one's by 3 dB {', '.join(str(x) for x in sorted(ms))} ms "
          f"before the end")  # fmt: skip

print("\n## 4. Shorter windows (from the 20 ms frames), dB\n")
print("| variant | last 0.4 s | last 0.2 s | last 0.1 s |")
print("|---|---|---|---|")
for v in ("masked", "silence"):
    cols = []
    for n in (20, 10, 5):
        x = sorted(window_margin(r[v]["tail_frames"], n) for r in by.values())
        cols.append(f"{x[0]:.1f} to {x[-1]:.1f}")
    print(f"| {v} | {' | '.join(cols)} |")
