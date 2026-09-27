"""Op-level #57608 probe on this build: [2, 8, T, 64] bf16, 1e30 planted in K and V implicit tile padding via
ttnn.fill_implicit_tile_padding. Configs: op default (no program config, as in #57608), HiFi4+fp32 acc, and ours
(SDPAProgramConfig q128/k256, exp_approx_mode=False, default compute config). PCC / max|diff| vs float32 torch."""
import torch, ttnn
from models.common.utility_functions import comp_pcc
device = ttnn.open_device(device_id=0, l1_small_size=65536)
try:
    grid = device.compute_with_storage_grid_size()
    ours = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=256, exp_approx_mode=False)
    ours_approx = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=256, exp_approx_mode=True)
    hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    configs = {"op default (as #57608)": dict(), "HiFi4 + fp32 acc": dict(compute_kernel_config=hifi4),
               "ours: q128/k256, exact exp": dict(program_config=ours), "q128/k256, approx exp": dict(program_config=ours_approx)}
    print("| T | config | clean PCC / max|diff| | K+V padding = 1e30 PCC / max|diff| |")
    print("|---|---|---|---|")
    for T in (161, 449, 448):
        torch.manual_seed(T)
        q = torch.randn(2, 8, T, 64) * 0.3; k = torch.randn(2, 8, T, 64); v = torch.randn(2, 8, T, 64)
        want = torch.softmax((q @ k.transpose(-1, -2)) / 8.0, -1) @ v
        dev = lambda a: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for name, kw in configs.items():
            res = []
            for plant in (False, True):
                qd, kd, vd = dev(q), dev(k), dev(v)
                if plant:
                    kd, vd = ttnn.fill_implicit_tile_padding(kd, 1e30), ttnn.fill_implicit_tile_padding(vd, 1e30)
                out = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(qd, kd, vd, is_causal=False, scale=1 / 8.0, **kw)).float()
                out = out[..., :T, :64]
                res.append((float(comp_pcc(want, out, 0.99)[1]), (want - out).abs().max().item()))
            print(f"| {T} | {name} | {res[0][0]:.5f} / {res[0][1]:.3g} | {res[1][0]:.5f} / {res[1][1]:.3g} |", flush=True)
finally:
    ttnn.close_device(device)
