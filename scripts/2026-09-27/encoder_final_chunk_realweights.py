"""Real flow.pt encoder: bucketed (fixed) vs exact-length TT, chunk-only control, and TT vs the torch
chunk-causal reference, at non-aligned final-chunk lengths. Large random values in the padded rows."""
import torch, ttnn
import models.demos.audio.cosyvoice2.tt.flow.encoder as em
from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.flow.encoder import (CHUNK_SIZE, CHUNK_SIZE_UP, TtUpsampleConformerEncoder,
    UpsampleConformerEncoderRef, bucket_length, chunk_causal_bias_torch)

def stats(w, g, lo, hi):
    w, g = w[:, lo:hi], g[:, lo:hi]
    return float(comp_pcc(w, g, 0.99)[1]), (w - g).abs().max().item()

flow_sd = load_checkpoint_file("flow.pt")
ref = UpsampleConformerEncoderRef.from_checkpoint(sub_state_dict(flow_sd, "encoder.")); ref.eval()
emb = flow_sd["input_embedding.weight"].float()
device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    tt = TtUpsampleConformerEncoder(device, ref)
    orig_bias = em.streaming_attn_bias_torch
    tt._bucket_padding = lambda b, r: ttnn.from_torch(torch.randn(b, r, 512) * 100.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    def run(t_len, x, **kw):
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.to_torch(tt(xd, t_len, 1, streaming=True, **kw)).float()
    print("| valid | bucket | case | valid frames PCC / max|diff| | last partial chunk PCC / max|diff| |")
    print("|---|---|---|---|---|")
    for valid in (70, 110, 137, 263):
        torch.manual_seed(valid)
        xs = emb[torch.randint(0, emb.shape[0], (1, valid))]
        bucket = bucket_length(valid); v2 = 2 * valid; lo = (v2 // CHUNK_SIZE_UP) * CHUNK_SIZE_UP
        padded = torch.cat([xs, torch.zeros(1, bucket - valid, 512)], dim=1)
        with torch.no_grad():
            want = ref(xs, streaming=True)
        exact = run(valid, xs)
        bucketed = run(bucket, padded, valid_length=valid)
        em.streaming_attn_bias_torch = lambda size, v, cs, neg=-30000.0: chunk_causal_bias_torch(size, cs, neg)
        chunk_only = run(bucket, padded, valid_length=valid)
        em.streaming_attn_bias_torch = orig_bias
        for name, a, b in (("TT exact vs torch ref", want, exact), ("TT bucketed vs torch ref", want, bucketed),
                           ("TT bucketed vs TT exact", exact, bucketed), ("CONTROL chunk-only vs TT exact", exact, chunk_only),
                           ("CONTROL chunk-only vs torch ref", want, chunk_only)):
            w, r = stats(a, b, 0, v2), stats(a, b, lo, v2)
            print(f"| {valid} | {bucket} | {name} | {w[0]:.6f} / {w[1]:.4g} | [{lo},{v2}) {r[0]:.6f} / {r[1]:.4g} |", flush=True)
        print(f"  (torch ref output: max|x| {want[:, :v2].abs().max().item():.3g}, mean|x| {want[:, :v2].abs().mean().item():.3g})", flush=True)
finally:
    ttnn.close_device(device)
