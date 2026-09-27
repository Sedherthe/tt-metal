"""O6: F0 predictor math fidelity A/B (HiFi4, today's config, vs HiFi3), real hift.pt weights, fp32 and bf16,
on real speech mels (LibriSpeech dummy validation 0/3/5), against the fp32 torch F0. Per arm, over voiced frames
(torch f0 > 10 Hz): mean/max |df| Hz, cents mean/p99/max, voiced/unvoiced flips, and the accumulated phase drift of
the fundamental, max |sum(df) * 0.02 s| in cycles (what breaks waveform PCC; phase_i = 2 pi sum f_k * 480/24000)."""
import librosa, numpy as np, torch, ttnn
from datasets import load_dataset
from librosa.filters import mel as librosa_mel_fn
import models.demos.audio.cosyvoice2.tt.hifigan.conv as conv
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef, TtConvRNNF0Predictor

def mel_spectrogram(y, n_fft=1920, num_mels=80, sr=24000, hop=480, win=1920, fmin=0, fmax=8000):
    basis = torch.from_numpy(librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
    y = torch.nn.functional.pad(y.unsqueeze(1), ((n_fft - hop) // 2, (n_fft - hop) // 2), mode="reflect").squeeze(1)
    spec = torch.view_as_real(torch.stft(y, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win),
                                         center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True))
    return torch.log(torch.clamp(basis @ torch.sqrt(spec.pow(2).sum(-1) + 1e-9), min=1e-5))

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
mels = []
for i in (0, 3, 5):
    a = ds[i]["audio"]; w = librosa.resample(np.asarray(a["array"], dtype=np.float32), orig_sr=a["sampling_rate"], target_sr=24000)
    mels.append(mel_spectrogram(torch.from_numpy(w).unsqueeze(0)).transpose(1, 2).contiguous())  # [1, T, 80]
ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(load_checkpoint_file("hift.pt"), "f0_predictor."))
with torch.no_grad():
    f0_ref = [ref(m.transpose(1, 2)).reshape(-1) for m in mels]

orig = (conv.accurate_compute_config, conv.safe_compute_config)
def with_fidelity(fid):
    def acc(device):
        return ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    def safe(device):
        return ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
    return acc, safe

device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    print("| arm | utterances (frames) | voiced | mean abs df Hz | max abs df Hz | cents mean / p99 / max | V/UV flips | max phase drift, cycles |")
    print("|---|---|---|---|---|---|---|---|")
    for fid_name, fid in (("HiFi4", ttnn.MathFidelity.HiFi4), ("HiFi3", ttnn.MathFidelity.HiFi3)):
        conv.accurate_compute_config, conv.safe_compute_config = with_fidelity(fid)
        for dt_name, dt in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
            tt = TtConvRNNF0Predictor(device, ref, dtype=dt)
            dfs, cents, flips, drift, nv = [], [], 0, 0.0, 0
            for m, fr in zip(mels, f0_ref):
                md = ttnn.from_torch(m, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
                ft = ttnn.to_torch(tt(md, m.shape[1], 1)).float().reshape(-1)[: fr.numel()]
                v = fr > 10.0
                flips += int(((ft > 10.0) != v).sum()); nv += int(v.sum())
                d = (ft - fr)[v]; dfs.append(d.abs())
                cents.append((1200 * torch.log2(ft[v].clamp(min=1e-3) / fr[v])).abs())
                drift = max(drift, (torch.cumsum((ft - fr) * v, 0) * 0.02).abs().max().item())
            dfs, cents = torch.cat(dfs), torch.cat(cents)
            print(f"| {fid_name} {dt_name} | 3 ({', '.join(str(m.shape[1]) for m in mels)}) | {nv} | {dfs.mean().item():.3f} | {dfs.max().item():.2f} | "
                  f"{cents.mean().item():.2f} / {torch.quantile(cents, 0.99).item():.1f} / {cents.max().item():.1f} | {flips} | {drift:.3f} |", flush=True)
finally:
    conv.accurate_compute_config, conv.safe_compute_config = orig
    ttnn.close_device(device)
