"""#57608 exposure check: non-streaming CFM estimator (real flow.pt weights), single forward, B=2 (as under CFG),
at T = 1 mod 32 across the realistic mel range, plus aligned T for comparison. Arms: fused SDPA (default),
explicit chain (COSYVOICE2_FLOW_SDPA=0), and a negative control (fused SDPA with 1e30 planted in K and V's
implicit tile padding). Each compared with the fp32 torch reference: PCC and max|diff|."""
import os, sys, time
import torch, ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
import models.demos.audio.cosyvoice2.tt.flow.decoder as dm

T_LIST = [int(a) for a in sys.argv[1:]] or [161, 289, 449, 577, 705, 833, 1153, 1537, 448, 832]
flow_sd = load_checkpoint_file("flow.pt")
ref = dm.CausalConditionalDecoderRef.from_checkpoint(sub_state_dict(flow_sd, "decoder.estimator.")); ref.eval()
device = ttnn.open_device(device_id=0, l1_small_size=65536)
real_sdpa = ttnn.transformer.scaled_dot_product_attention
def planted_sdpa(q, k, v, **kw):
    return real_sdpa(q, ttnn.fill_implicit_tile_padding(k, 1e30), ttnn.fill_implicit_tile_padding(v, 1e30), **kw)
try:
    os.environ["COSYVOICE2_FLOW_SDPA"] = "1"; tt_fused = dm.TtCausalConditionalDecoder(device, ref)
    os.environ["COSYVOICE2_FLOW_SDPA"] = "0"; tt_chain = dm.TtCausalConditionalDecoder(device, ref)
    os.environ["COSYVOICE2_FLOW_SDPA"] = "1"
    print("| T | T mod 32 | fused SDPA vs torch: PCC / max|diff| | explicit chain vs torch | CONTROL fused + planted K/V padding | torch max|out| |")
    print("|---|---|---|---|---|---|")
    for T in T_LIST:
        torch.manual_seed(T)
        b = 2
        x = torch.randn(b, T, 80); mu = torch.randn(b, T, 80) * 0.5; mu[1] = 0
        cond = torch.zeros(b, T, 80); cond[0, : T // 3] = torch.randn(T // 3, 80) * 0.5
        spks = torch.randn(b, 80) * 0.3; spks[1] = 0
        mask = torch.ones(b, T, 1); t = torch.rand(1).repeat(b)
        with torch.no_grad():
            want = ref(x, mask, mu, t, spks, cond)
        dev = lambda a: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        args = (dev(x), dev(mask), dev(mu), t, dev(spks.unsqueeze(1)), dev(cond), T, b)
        def run(dec):
            return ttnn.to_torch(dec(*args)).float().reshape(want.shape)
        t0 = time.time(); fused = run(tt_fused); tf = time.time() - t0
        chain = run(tt_chain)
        ttnn.transformer.scaled_dot_product_attention = planted_sdpa
        try:
            control = run(tt_fused)
        finally:
            ttnn.transformer.scaled_dot_product_attention = real_sdpa
        def st(got):
            return float(comp_pcc(want, got, 0.99)[1]), (want - got).abs().max().item()
        f, c, n = st(fused), st(chain), st(control)
        print(f"| {T} | {T % 32} | {f[0]:.6f} / {f[1]:.4g} | {c[0]:.6f} / {c[1]:.4g} | {n[0]:.6f} / {n[1]:.4g} | {want.abs().max().item():.3g} |", flush=True)
finally:
    ttnn.close_device(device)
