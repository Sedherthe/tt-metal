"""For the chunked-HiFT gate's own-F0 figures: how far is TT's SINGLE-pass HiFT from upstream's single pass on the
same three reference mels, same noise? If TT single vs upstream single is ~0.1 log-mel L1 too, that level is the
port's own-F0 error and chunking adds nothing to it. Also TT chunked vs TT single (TT's own chunking effect)."""
import glob
import json
import os

import numpy as np
import torch

import ttnn
from models.experimental.cosyvoice2.tests.pcc.test_hift_chunked import _logmel
from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.experimental.cosyvoice2.tt.hifigan.conv import config_tensors_in_dram_override
from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.experimental.cosyvoice2.tt.hifigan.generator import (
    TorchHiFTDecodeRef,
    TorchHiFTGeneratorInferenceRef,
    TtHiFTDecoder,
    TtHiFTGenerator,
)

REF = "/home/user/data/cosyvoice2_hift_stream_ref"
device = ttnn.open_device(device_id=0, l1_small_size=65536)
rows = []
try:
    with config_tensors_in_dram_override(True):
        hift_sd = load_checkpoint_file("hift.pt")
        decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
        f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
        ref = TorchHiFTGeneratorInferenceRef(decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"],
                                             hift_sd["m_source.l_linear.bias"])
        gen = TtHiFTGenerator(device, ref, TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32), dtype=ttnn.float32)
    for path in sorted(glob.glob(os.path.join(REF, "*.npz"))):
        d = np.load(path)
        mel, noise = torch.from_numpy(d["mel"]), torch.from_numpy(d["noise"])
        frames = mel.shape[1]
        up_single, up_chunked = torch.from_numpy(d["single"]), torch.from_numpy(d["chunked"])
        mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_single = ttnn.to_torch(gen.inference(mel_dev, frames, 1, sine_noise=noise)).float().reshape(-1)[: frames * 480]
        with torch.no_grad():
            f0_full = f0_ref(mel.transpose(1, 2)).reshape(1, frames)  # torch F0 over the whole mel, for injection
        tt_single_inj = ttnn.to_torch(gen.inference(mel_dev, frames, 1, sine_noise=noise, f0=f0_full)).float().reshape(-1)[: frames * 480]
        ttnn.deallocate(mel_dev)
        tt_chunked = gen.inference_chunked(mel, noise)
        lm = {k: _logmel(v) for k, v in dict(up_single=up_single, up_chunked=up_chunked, tt_single=tt_single,
                                                   tt_single_inj=tt_single_inj, tt_chunked=tt_chunked).items()}
        l1 = lambda a, b: round(float((lm[a] - lm[b]).abs().mean()), 4)  # noqa: E731
        row = {
            "case": os.path.basename(path)[:-4], "frames": frames,
            "tt_single_own_f0 vs up_single": l1("tt_single", "up_single"),
            "tt_single_torch_f0 vs up_single": l1("tt_single_inj", "up_single"),
            "tt_chunked_own_f0 vs up_chunked": l1("tt_chunked", "up_chunked"),
            "tt_chunked vs tt_single (TT's chunking effect)": l1("tt_chunked", "tt_single"),
            "up_chunked vs up_single (upstream's chunking effect)": l1("up_chunked", "up_single"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
finally:
    ttnn.close_device(device)
