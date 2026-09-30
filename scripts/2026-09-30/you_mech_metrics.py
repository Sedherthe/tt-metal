"""The mechanism outputs of you_tail_ab.py / you_tail_front.py against upstream's streamed audio, one table for every
variant: upstream's mel, F0 and noise went in, so only the final HiFT call's padding differs from upstream.

Per case and variant: the final seam's PCC (the final chunk's 3,840-sample crossfade, 480 samples either side); the
final chunk's PCC before its last 0.4 s; in the last 0.4 s, the difference's level against the signal's; the last
20 ms against upstream's; the near-silent samples (|x| < 1e-4) at the very end.

    python you_mech_metrics.py <tail_ab dir>
"""
import os
import sys

import numpy as np
import soundfile

REF = "/home/user/data/cosyvoice2_streaming_ref"
SR = 24000
SOURCE_CACHE = 3840


def dbfs(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-12))


def pcc(a, b):
    return float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1])


def main() -> None:
    d = sys.argv[1]
    cases = sorted(p[: -len(".npz")] for p in os.listdir(REF) if p.startswith("zero_shot_") and p.endswith(".npz"))
    print("| case | variant | final seam PCC | final chunk PCC before its last 0.4 s | last 0.4 s: signal / difference, dBFS "
          "| last 20 ms, ours / upstream, dBFS | quiet samples at the end |")
    print("|---|---|---|---|---|---|---|")
    for c in cases:
        ref = np.load(f"{REF}/{c}.npz")
        up = ref["audio"]
        k = len(ref["offsets"]) - 1
        start = sum(len(ref[f"speech_{i}"]) for i in range(k))
        n4, n1 = int(0.4 * SR), int(0.02 * SR)
        for v in ("padded", "exact", "front"):
            p = f"{d}/mech/{v}_{c}.wav"
            if not os.path.exists(p):
                continue
            m = soundfile.read(p, dtype="float32")[0]
            assert len(m) == len(up), (p, len(m), len(up))
            seam = pcc(m[start - 480 : start + SOURCE_CACHE + 480], up[start - 480 : start + SOURCE_CACHE + 480])
            body = slice(start, len(up) - n4)
            body_pcc = f"{pcc(m[body], up[body]):.5f}" if body.stop - body.start > 960 else "(chunk within its tail)"
            big = np.nonzero(np.abs(m) >= 1e-4)[0]
            quiet = len(m) - 1 - big[-1] if len(big) else len(m)
            print(f"| {c[10:]} | {v} | {seam:.5f} | {body_pcc} | {dbfs(up[-n4:]):.1f} / {dbfs(m[-n4:] - up[-n4:]):.1f} "
                  f"| {dbfs(m[-n1:]):.1f} / {dbfs(up[-n1:]):.1f} | {quiet} |")


if __name__ == "__main__":
    main()
