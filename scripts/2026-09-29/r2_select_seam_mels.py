"""R2: choose the seam gate's six mels, one per speaker, cut so that every seam lands in voiced speech (9 seams).

RUN IN THE REFERENCE VENV (upstream's frontend and HiFT F0 predictor), from /tmp:

    env -u PYTHONPATH COSYVOICE2_REPO=... LIBRISPEECH_ROOT=... HF_HOME=... $COSYVOICE2_REF_ENV/bin/python \\
        r2_select_seam_mels.py > r2_select_seam_mels.log

The rule, deterministic from the data:
- Six cases, lengths 600, 800 and 1016 frames (two calls: one seam) and 1100, 1300 and 1520 (three calls: two seams).
  The schedules cover an anchored last call with a long overlap, one with a short overlap, and calls that line up.
  Their genders alternate F, M, F, M, F, M.
- A seam's crossfade covers mel frames [504, 512) of the cut (and [1008, 1016) for the second seam) whatever the
  length (tt/hifigan/chunking.py).
- For each case in turn: take the lowest-numbered test-clean speaker of that gender not used yet. Within it, take
  utterances in id order, and the smallest start offset (in frames) at which upstream's own F0 predictor marks
  every frame of each crossfade, +-4 frames, as voiced (F0 > 10 Hz, upstream's `nsf_voiced_threshold`). Move on to
  the next speaker if none fits.
"""
import glob
import json
import os
import sys

import soundfile

sys.path.insert(0, "/home/user/tt-metal/models/experimental/cosyvoice2/scripts")
import reference_env  # noqa: E402

LENGTHS = [600, 800, 1016, 1100, 1300, 1520]
GENDERS = ["F", "M", "F", "M", "F", "M"]
SEAMS = {2: [504], 3: [504, 1008]}  # calls -> crossfade start frames within the cut
MARGIN, CROSSFADE, VOICED_HZ, FPS = 4, 8, 10.0, 50


def calls(frames):
    return 2 if frames <= 1016 else 3


def speakers():
    root = reference_env.librispeech_root()
    out = []
    for line in open(os.path.join(root, "LibriSpeech", "SPEAKERS.TXT")):
        if line.startswith(";"):
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 3 and f[2] == "test-clean":
            out.append((int(f[0]), f[1]))
    return sorted(out)


def main():
    import torch

    model = reference_env.load_upstream()
    hift = model.model.hift
    root = os.path.join(reference_env.librispeech_root(), "LibriSpeech", "test-clean")
    used, chosen = set(), []
    for frames, gender in zip(LENGTHS, GENDERS):
        seams = SEAMS[calls(frames)]
        found = None
        for spk, g in speakers():
            if g != gender or spk in used or found:
                continue
            for path in sorted(glob.glob(os.path.join(root, str(spk), "*", "*.flac"))):
                if soundfile.info(path).duration * FPS < frames + 1:
                    continue
                feat, _ = model.frontend._extract_speech_feat(path)  # [1, T, 80]
                with torch.inference_mode():
                    f0 = hift.f0_predictor(feat.transpose(1, 2).float()).reshape(-1)  # [T]
                voiced = (f0 > VOICED_HZ).tolist()
                for off in range(0, feat.shape[1] - frames + 1):
                    if all(all(voiced[off + s - MARGIN : off + s + CROSSFADE + MARGIN]) for s in seams):
                        utt = os.path.basename(path)[: -len(".flac")]
                        found = {
                            "utt": utt, "speaker": spk, "gender": gender, "start": off, "frames": frames,
                            "calls": calls(frames), "utterance_frames": int(feat.shape[1]),
                            "seam_f0_min_hz": [round(float(f0[off + s : off + s + CROSSFADE].min()), 1) for s in seams],
                        }
                        break
                if found:
                    break
            if found:
                used.add(spk)
        assert found, f"no {gender} speaker fits a {frames}-frame cut"
        chosen.append(found)
        print(json.dumps(found), flush=True)
    print("\nCASES = {  # utterance id -> (first mel frame, mel frames)")
    for c in chosen:
        print(f'    "{c["utt"]}": ({c["start"]}, {c["frames"]}),  # {c["gender"]}, {c["calls"]} calls')
    print("}")


if __name__ == "__main__":
    main()
