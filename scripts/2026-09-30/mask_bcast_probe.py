"""Device probe for B28's fix: does a 0/1 mask multiply broadcast, and exactly, on the shapes HiFT uses?
- [1, L, C] fp32 TILE (DRAM, like a conv output reshaped to [N, L, C]) x [1, L, 1] mask -> broadcast over channels;
- [1, 9, T] x [1, 1, T] (the iSTFT magnitude, bins x frames) -> broadcast over bins.
Exact means: equal to the torch product bit for bit (x * 1.0 and x * 0.0)."""
import torch

import ttnn

device = ttnn.open_device(device_id=0, l1_small_size=65536)
try:
    torch.manual_seed(0)
    for L, C, valid in ((1016, 512, 1000), (8129, 64, 8000), (30721, 64, 30001)):
        x = torch.randn(1, L, C)
        m = torch.zeros(1, L, 1)
        m[:, :valid] = 1.0
        xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        mt = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(ttnn.multiply(xt, mt)).float()
        print(f"[1,{L},{C}] x [1,{L},1]: shape {tuple(got.shape)}, exact {torch.equal(got, x * m)}, "
              f"tail zero {bool((got[:, valid:] == 0).all())}")
    T = 30721
    x = torch.rand(1, 9, T) * 3
    m = torch.zeros(1, 1, T)
    m[..., :30001] = 1.0
    got = ttnn.to_torch(ttnn.multiply(ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device),
                                      ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))).float()
    print(f"[1,9,{T}] x [1,1,{T}]: shape {tuple(got.shape)}, exact {torch.equal(got, x * m)}")
finally:
    ttnn.close_device(device)
