"""RUNBOOK section 1 smoke test: a 64x64 ttnn.add with a to_torch read-back."""
import time

import torch
import ttnn

t0 = time.time()
device = ttnn.open_device(device_id=0)
print(f"opened device in {time.time() - t0:.1f} s; arch {device.arch()}, grid {device.compute_with_storage_grid_size()}")
torch.manual_seed(0)
a = torch.rand(64, 64, dtype=torch.bfloat16)
b = torch.rand(64, 64, dtype=torch.bfloat16)
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.to_torch(ttnn.add(ta, tb))
ref = a + b
diff = (out.float() - ref.float()).abs()
raw = out.view(torch.int16)
print(f"shape {tuple(out.shape)}, max|diff| {diff.max().item():.3g}, finite {bool(torch.isfinite(out).all())}, "
      f"all-ones bit pattern present {bool((raw == -1).any())}, bit-identical {torch.equal(out, ref)}")
ttnn.close_device(device)
print(f"closed; total {time.time() - t0:.1f} s")
assert torch.isfinite(out).all() and diff.max().item() <= 2 ** -6, "read-back does not match"
print("SMOKE OK")
