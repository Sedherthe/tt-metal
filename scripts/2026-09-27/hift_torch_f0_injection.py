"""Finding 9: is HiFT's own-F0 waveform gap F0 phase drift? Real hift.pt, fp32 throughout (the 2026-09-24 setup
that gave PCC 0.977 at mel_frames=16), a real speech mel (LibriSpeech dummy ds[3], resampled to 24 kHz, upstream
mel settings). Arms vs the fp32 torch reference, shared sine_noise: (a) TT with its own F0; (b) TT with the torch
F0 injected into the NSF source (f0 -> repeat_interleave -> source -> decode, as TtHiFTGenerator.inference does)."""
import sys
import librosa, numpy as np, torch, ttnn
from datasets import load_dataset
from librosa.filters import mel as librosa_mel_fn
from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import (TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef,
    TtHiFTDecoder, TtHiFTGenerator)

def mel_spectrogram(y, n_fft=1920, num_mels=80, sr=24000, hop=480, win=1920, fmin=0, fmax=8000):  # cv2_frontend's, verbatim math
    basis = torch.from_numpy(librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
    y = torch.nn.functional.pad(y.unsqueeze(1), ((n_fft - hop) // 2, (n_fft - hop) // 2), mode="reflect").squeeze(1)
    spec = torch.view_as_real(torch.stft(y, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win),
                                         center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True))
    return torch.log(torch.clamp(basis @ torch.sqrt(spec.pow(2).sum(-1) + 1e-9), min=1e-5))

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
a = ds[3]["audio"]; wav24 = librosa.resample(np.asarray(a["array"], dtype=np.float32), orig_sr=a["sampling_rate"], target_sr=24000)
mel_full = mel_spectrogram(torch.from_numpy(wav24).unsqueeze(0)).transpose(1, 2)  # [1, T, 80]

hift_sd = load_checkpoint_file("hift.pt")
decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
ref = TorchHiFTGeneratorInferenceRef(decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"])
with torch.no_grad():
    f0_full = f0_ref(mel_full.transpose(1, 2)).reshape(-1)

def window(n):  # the n-frame window with the most voiced frames (f0 > 10 Hz threshold)
    voiced = (f0_full > 10).float()
    best = max(range(0, mel_full.shape[1] - n + 1), key=lambda i: (voiced[i:i + n].sum().item(), -i))
    return best

device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    dec = TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32)
    gen = TtHiFTGenerator(device, ref, dec, dtype=ttnn.float32)
    print(f"utterance: {mel_full.shape[1]} mel frames")
    print("| mel_frames | window | voiced | F0 max|dTT-torch| Hz | arm | PCC | max|diff| | whole-wave max|ref| |")
    print("|---|---|---|---|---|---|---|---|")
    for n in [int(x) for x in sys.argv[1:]] or [16, 108, 208]:
        i0 = window(n); mel = mel_full[:, i0:i0 + n].contiguous()
        torch.manual_seed(n); sine_noise = torch.randn(1, n * ref.upsample_scale, ref.harmonic_num + 1)
        with torch.no_grad():
            want = ref.inference(mel, sine_noise=sine_noise).reshape(1, -1)
            f0_t = f0_ref(mel.transpose(1, 2)).reshape(1, n)
        mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        f0_tt = ttnn.to_torch(gen.f0_predictor(mel_dev, n, 1)).float().reshape(1, n)
        own = ttnn.to_torch(gen.inference(mel_dev, n, 1, sine_noise=sine_noise)).reshape(1, -1).float()
        f0_dev = ttnn.from_torch(f0_t.reshape(1, n, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        f0_audio = ttnn.repeat_interleave(f0_dev, gen.upsample_scale, dim=1)
        noise_dev = ttnn.from_torch(sine_noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        s, _, _ = gen.source(f0_audio, sine_noise=noise_dev)
        inj = ttnn.to_torch(gen.decoder.decode(mel_dev, s, n, 1)).reshape(1, -1).float()
        df0 = (f0_tt - f0_t).abs().max().item(); nv = int((f0_t > 10).sum())
        for arm, got in (("own F0", own), ("torch F0 injected", inj)):
            pcc = float(comp_pcc(want, got, 0.99)[1])
            print(f"| {n} | [{i0},{i0 + n}) | {nv}/{n} | {df0:.3g} | {arm} | {pcc:.6f} | {(want - got).abs().max().item():.4g} | {want.abs().max().item():.3g} |", flush=True)
finally:
    ttnn.close_device(device)
