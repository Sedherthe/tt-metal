"""The "you" clip, audio side (CPU, no device): lengths for all six utterances, and the tails.

Lengths: our streamed audio (R5 live, R3 offline) and our Stage 1 audio against upstream's streamed audio of the same
tokens, whole and per chunk (upstream's `speech_k` against R5's per-chunk records).

Tails, for every case: the last second of each version, in 20 ms frames, where speech ends (the last frame within
35 dB of the utterance's loudest), the level after that, the residual's spectrum by band, and the last 10 ms. For
260-123440-0010 the tails are written to ~/listening/tails/ (the last 1 s, and the last 2.5 s for context).

    python you_lengths_tails.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import soundfile
from scipy.signal import welch

D = "/home/user/data"
SR = 24000
CASES = ["zero_shot_121-127105-0003", "zero_shot_121-127105-0015", "zero_shot_121-127105-0024",
         "zero_shot_260-123286-0014", "zero_shot_260-123440-0002", "zero_shot_260-123440-0010"]  # fmt: skip
SRC = {
    "upstream_stream": f"{D}/cosyvoice2_streaming_ref",
    "ours_live": f"{D}/cosyvoice2_runs/0929/r5_stream_warm1",
    "ours_offline": f"{D}/cosyvoice2_runs/0929/stream_offline",
    "ours_stage1": f"{D}/cosyvoice2_runs/0929/stage1_r1",
}
YOU = "zero_shot_260-123440-0010"
LISTEN = os.path.expanduser("~/listening/tails")
FRAME = int(0.02 * SR)
BANDS = ((0, 500), (500, 2000), (2000, 5000), (5000, 12000))


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def frames_db(a: np.ndarray) -> np.ndarray:
    n = len(a) // FRAME
    return np.array([db(np.sqrt(np.mean(a[i * FRAME:(i + 1) * FRAME] ** 2))) for i in range(n)])


def speech_end_s(a: np.ndarray) -> float:
    f = frames_db(a)
    return (np.nonzero(f >= f.max() - 35.0)[0][-1] + 1) * FRAME / SR


def band_db(seg: np.ndarray) -> list[float]:
    if len(seg) < 256:
        return [float("nan")] * len(BANDS)
    f, p = welch(seg, SR, nperseg=min(512, len(seg)))
    return [round(10 * np.log10(max(p[(f >= lo) & (f < hi)].sum() * (f[1] - f[0]), 1e-24)), 1) for lo, hi in BANDS]


def main() -> None:
    wav = {(c, s): soundfile.read(f"{d}/{c}.wav", dtype="float32")[0] for c in CASES for s, d in SRC.items()}
    live = {r["case_id"]: r for r in json.load(open(f"{SRC['ours_live']}/results.json"))["results"]}

    print("## Lengths (samples at 24 kHz)\n")
    print("| case | upstream streamed | ours live (R5) | ours offline (R3) | ours Stage 1 | chunks: upstream / ours live |")
    print("|---|---|---|---|---|---|")
    for c in CASES:
        z = np.load(f"{SRC['upstream_stream']}/{c}.npz")
        up_chunks = [len(z[f"speech_{k}"]) for k in range(len(z["offsets"]))]
        our_chunks = [int(round(ch["audio_s"] * SR)) for ch in live[c]["chunks"]]
        lens = [len(wav[(c, s)]) for s in SRC]
        same = "same" if up_chunks == our_chunks else "DIFFER"
        print(f"| {c} | {lens[0]} | {lens[1]} | {lens[2]} | {lens[3]} | {same}: {up_chunks} / {our_chunks} |")

    print("\n## Tails\n")
    print("speech end: the last 20 ms frame within 35 dB of the utterance's loudest. after: level from there to the end.")
    print("bands: the residual after speech end, dB, for 0-0.5 / 0.5-2 / 2-5 / 5-12 kHz. diff: ours - upstream.\n")
    print("| case | version | speech end s | after, dBFS | last 0.4 s, dBFS | diff last 0.4 s, dBFS | last 10 ms peak | bands after |")
    print("|---|---|---|---|---|---|---|---|")
    for c in CASES:
        up = wav[(c, "upstream_stream")]
        for s in SRC:
            a = wav[(c, s)]
            end = speech_end_s(a)
            after = a[int(end * SR):]
            n4 = int(0.4 * SR)
            diff = "" if s == "upstream_stream" else f"{db(np.sqrt(np.mean((a[-n4:] - up[-n4:]) ** 2))):.1f}"
            print(f"| {c} | {s} | {end:.2f} of {len(a) / SR:.2f} | {db(np.sqrt(np.mean(after ** 2))) if len(after) else float('nan'):.1f} | "
                  f"{db(np.sqrt(np.mean(a[-n4:] ** 2))):.1f} | {diff} | {np.abs(a[-240:]).max():.4f} | {band_db(after)} |")  # fmt: skip

    print(f"\n## {YOU}: the last second in 20 ms frames, dBFS\n")
    n = SR
    print("| t from end, s | " + " | ".join(SRC) + " |")
    print("|---|" + "---|" * len(SRC))
    env = {s: frames_db(wav[(YOU, s)][-n:]) for s in SRC}
    for i in range(len(env["upstream_stream"])):
        t = (n - i * FRAME) / SR
        print(f"| -{t:.2f} | " + " | ".join(f"{env[s][i]:.1f}" for s in SRC) + " |")

    os.makedirs(LISTEN, exist_ok=True)
    for s in SRC:
        a = wav[(YOU, s)]
        soundfile.write(f"{LISTEN}/{YOU}_{s}_last1s.wav", a[-SR:], SR)
        soundfile.write(f"{LISTEN}/{YOU}_{s}_last2.5s.wav", a[-int(2.5 * SR):], SR)
    print(f"\nwrote {LISTEN}: {sorted(os.listdir(LISTEN))}")


if __name__ == "__main__":
    main()
