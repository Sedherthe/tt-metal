"""Stage 3 plan, step 1: which utterances stream at RTF above 1.0, and why. From the demo's per-chunk records (the
masked-HiFT runs of 09-30, `demo.py --stream`, two fresh processes), with the Stage 1 demo's same utterances beside.

Each streamed utterance's wall time splits into text and LLM (prefill and decode, the chunks' work excluded), each
chunk's flow and HiFT (run between decode steps, or after the LLM for the final chunk), and the rest. RTF is wall
over audio, so each part's share of RTF is its time over the audio.

1. Per utterance: that split, and the RTF the flows alone would give.
2. Per chunk: the tokens its flow covers (the prompt, every token so far, and 3 look-ahead; the final chunk every
   token, non-streaming), the flow bucket that puts it at (`bucket_for`, as `tt/streaming.py` picks it), and its
   flow, CFM, HiFT and emitted audio.
3. The flow's and the CFM's cost per bucket, over both runs.

    /opt/venv/bin/python rtf_breakdown.py > rtf_breakdown.md    # imports the pipeline's bucket rule; no device
"""
import json
from collections import defaultdict

import numpy as np

from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, bucket_for

RUNS = {"run 1": "/home/user/data/cosyvoice2_runs/0930/stream_masked1", "run 2": "/home/user/data/cosyvoice2_runs/0930/stream_masked2"}
STAGE1 = "/home/user/data/cosyvoice2_runs/0930/stage1_masked"
INPUTS = "/home/user/data/cosyvoice2_inputs"
LOOKAHEAD = 3
BUCKETS = CosyVoice2Config.reported().flow_token_buckets()


def load(d):
    return {r["case_id"]: r for r in json.load(open(f"{d}/results.json"))["results"]}


def prompt_tokens(case_id):
    return int(np.load(f"{INPUTS}/{case_id}.npz")["flow_prompt_speech_tokens"].shape[-1])


stage1 = load(STAGE1)
print("## 1. Per utterance\n")
print("| case | run | audio s | tokens | chunks (tokens each) | text + LLM s | flows s (CFM) | HiFT s | rest s | wall s | "
      "RTF | RTF from the flows | Stage 1 RTF |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
per_bucket = defaultdict(lambda: {"flow": [], "cfm": [], "rest": []})
chunk_rows = []
for name, d in RUNS.items():
    for cid, r in sorted(load(d).items()):
        st, chunks = r["stage_s"], r["chunks"]
        llm = st["llm_prefill"] + st["llm_decode"]
        flows = [c["flow"] for c in chunks]
        cfm = sum(c["cfm"] for c in chunks)
        hift = sum(c["hift"] for c in chunks)
        wall = r["wall_s"]
        rest = wall - llm - sum(flows) - hift
        hops = "+".join(str(c["hop"]) for c in chunks)
        n_tok = sum(len(t) for t in r["segment_tokens"])
        print(f"| {cid[10:]} | {name} | {r['audio_s']:.2f} | {n_tok} | {len(chunks)} ({hops}) | "
              f"{llm:.2f} | {sum(flows):.2f} ({cfm:.2f}): {' + '.join(f'{f:.2f}' for f in flows)} | {hift:.2f} | {rest:.2f} | "
              f"{wall:.2f} | **{r['rtf']:.3f}** | {sum(flows) / r['audio_s']:.2f} | {stage1[cid]['rtf']:.3f} |")  # fmt: skip
        p = prompt_tokens(cid)
        for k, c in enumerate(chunks):
            covered = p + (n_tok if c["final"] else c["offset"] + c["hop"] + LOOKAHEAD)
            bucket = bucket_for(covered, BUCKETS)
            kind = "final, non-streaming" if c["final"] else "streaming"
            per_bucket[(bucket, kind)]["flow"].append(c["flow"])
            per_bucket[(bucket, kind)]["cfm"].append(c["cfm"])
            per_bucket[(bucket, kind)]["rest"].append(c["flow"] - c["cfm"])
            if name == "run 1":
                chunk_rows.append((cid[10:], k, c["hop"], kind, covered, bucket, c["flow"], c["cfm"], c["hift"], c["audio_s"]))

print("\n## 2. Per chunk (run 1)\n")
print("| case | chunk | hop | flow | tokens the flow covers | flow bucket (mel frames) | flow s | CFM s | HiFT s | audio emitted s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for cid, k, hop, kind, covered, bucket, flow, cfm, hift, audio in chunk_rows:
    print(f"| {cid} | {k} | {hop} | {kind} | {covered} | {bucket} ({2 * bucket}) | {flow:.2f} | {cfm:.2f} | {hift:.2f} | {audio:.2f} |")

print("\n## 3. Cost per flow bucket, both runs\n")
print("| flow bucket (mel frames) | flow | chunks | flow s | CFM s | CFM per Euler step, ms | flow outside the CFM, s |")
print("|---|---|---|---|---|---|---|")
for (bucket, kind), v in sorted(per_bucket.items()):
    f, c, o = v["flow"], v["cfm"], v["rest"]
    print(f"| {bucket} ({2 * bucket}) | {kind} | {len(f)} | {min(f):.2f}–{max(f):.2f} | {min(c):.2f}–{max(c):.2f} | "
          f"{100 * min(c):.0f}–{100 * max(c):.0f} | {min(o):.2f}–{max(o):.2f} |")  # fmt: skip
