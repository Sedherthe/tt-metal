"""Deterministic selection of the token-accuracy extension from LibriSpeech test-clean (run once, result pasted
into scripts/corpus.py). The primary set's rule, continued:
- the two primary speakers (260 M, 121 F): six more targets each, nominal 4, 5, 6, 10, 11, 12 s;
- per gender, the next qualifying speaker after the primary one (the same rule: lowest-numbered with an utterance
  within 20 % of every nominal length): a ~7 s prompt and four targets, nominal 3, 6, 9, 12 s.
Within a speaker: the closest utterance to each nominal length within 20 %, ties broken by utterance id, never
reusing one (the primary prompt/targets included)."""
import glob
import json
import os

import soundfile

ROOT = "/home/user/data/LibriSpeech"
SUB = "test-clean"
PRIMARY = {"260": ("260-123286-0016", ["260-123286-0014", "260-123440-0010", "260-123440-0002"]),
           "121": ("121-121726-0003", ["121-127105-0015", "121-127105-0003", "121-127105-0024"])}


def speakers():
    out = {}
    for line in open(os.path.join(ROOT, "SPEAKERS.TXT")):
        if line.startswith(";"):
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 3 and f[2] == SUB:
            out[f[0]] = f[1]
    return out


def utterances(spk):
    utts = {}
    for trans in glob.glob(os.path.join(ROOT, SUB, spk, "*", "*.trans.txt")):
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            path = os.path.join(os.path.dirname(trans), f"{uid}.flac")
            info = soundfile.info(path)
            utts[uid] = (round(info.frames / info.samplerate, 3), text)
    return utts


def pick(utts, nominal, used):
    cands = [(abs(d - nominal), uid) for uid, (d, _) in utts.items() if uid not in used and abs(d - nominal) <= 0.2 * nominal]
    if not cands:
        return None
    return min(cands)[1]


def qualifies(utts, lengths):
    used = set()
    for n in lengths:
        u = pick(utts, n, used)
        if u is None:
            return False
        used.add(u)
    return True


spk_gender = speakers()
result = {"primary_speakers": {}, "new_speakers": {}}
for spk, (prompt, targets) in PRIMARY.items():
    utts = utterances(spk)
    used = {prompt, *targets}
    picks = []
    for n in (4, 5, 6, 10, 11, 12):
        u = pick(utts, n, used)
        assert u, (spk, n)
        used.add(u)
        picks.append((u, utts[u][0], utts[u][1]))
    result["primary_speakers"][spk] = picks

for gender, primary in (("M", "260"), ("F", "121")):
    for spk in sorted((s for s, g in spk_gender.items() if g == gender), key=int):
        if int(spk) <= int(primary):
            continue
        utts = utterances(spk)
        if not qualifies(utts, (7, 3, 6, 9, 12)):
            continue
        used = set()
        prompt = pick(utts, 7, used)
        used.add(prompt)
        picks = []
        for n in (3, 6, 9, 12):
            u = pick(utts, n, used)
            used.add(u)
            picks.append((u, utts[u][0], utts[u][1]))
        result["new_speakers"][spk] = {"gender": gender, "prompt": (prompt, utts[prompt][0], utts[prompt][1]), "targets": picks}
        break
print(json.dumps(result, indent=1))
