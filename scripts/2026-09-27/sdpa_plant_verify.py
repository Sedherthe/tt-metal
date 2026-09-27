import torch, ttnn
from models.common.utility_functions import comp_pcc
device = ttnn.open_device(device_id=0, l1_small_size=65536)
try:
    T, Tp = 449, 480
    torch.manual_seed(0)
    q = torch.randn(2, 8, T, 64) * 0.3; k = torch.randn(2, 8, T, 64); v = torch.randn(2, 8, T, 64)
    want = torch.softmax((q @ k.transpose(-1, -2)) / 8.0, -1) @ v
    dev = lambda a: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    def padding_rows(t):  # view the padded rows: reinterpret with logical == padded shape, no fill
        try:
            full = ttnn.reshape(t, ttnn.Shape([2, 8, Tp, 64]), ttnn.Shape([2, 8, Tp, 64]))
            return ttnn.to_torch(full).float()[:, :, T:, :]
        except Exception as e:
            return f"reshape-view failed: {type(e).__name__}: {str(e)[:120]}"
    # (a) fill_implicit_tile_padding
    ka = ttnn.fill_implicit_tile_padding(dev(k), 1e30); va = ttnn.fill_implicit_tile_padding(dev(v), 1e30)
    pa = padding_rows(ka)
    print("(a) fill_implicit_tile_padding: padding rows ->", pa if isinstance(pa, str) else f"min {pa.min().item():.3g} max {pa.max().item():.3g}")
    # (b) #57608's method: upload padded garbage, then reshape to logical with skip_padding_fill
    def planted_b(x):
        xp = torch.cat([x, torch.full((2, 8, Tp - T, 64), 1e30)], dim=2)
        t = dev(xp)
        try:
            return ttnn.reshape(t, ttnn.Shape([2, 8, T, 64]), ttnn.Shape([2, 8, Tp, 64]), skip_padding_fill=True), None
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:160]}"
    kb, eb = planted_b(k); vb, _ = planted_b(v)
    if eb:
        print("(b) reshape with skip_padding_fill not available:", eb)
    else:
        pb = padding_rows(kb)
        print("(b) skip_padding_fill reshape: logical", tuple(kb.shape), "padding rows ->", pb if isinstance(pb, str) else f"min {pb.min().item():.3g} max {pb.max().item():.3g}")
    for name, kk, vv in (("clean", dev(k), dev(v)), ("(a) fill_implicit", ka, va)) + ((("(b) skip_padding_fill", kb, vb),) if not eb else ()):
        out = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(dev(q), kk, vv, is_causal=False, scale=1 / 8.0)).float()[..., :T, :64]
        print(f"  SDPA op default, {name}: PCC {float(comp_pcc(want, out, 0.99)[1]):.5f} max|diff| {(want - out).abs().max().item():.4g}", flush=True)
finally:
    ttnn.close_device(device)
