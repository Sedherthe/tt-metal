"""tenstorrent/tt-metal#36487's own reproducer, verbatim except: deallocate_activation=False (both convs read the
same input), a PCC for the prepared output too, and min/max printed as floats. Run on this build (N150) to see
whether that bug reproduces here."""
import torch
import torch.nn as nn

import ttnn

# Setup
torch.manual_seed(42)
weight = torch.randn(64, 3, 7, 7, dtype=torch.bfloat16)
bias = torch.randn(64, dtype=torch.bfloat16)
torch.manual_seed(123)
input_tensor = torch.randn(1, 3, 800, 1280, dtype=torch.bfloat16)

# Torch reference
model = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True), nn.ReLU()).to(torch.bfloat16)
with torch.no_grad():
    model[0].weight.copy_(weight)
    model[0].bias.copy_(bias)
model.eval()
with torch.no_grad():
    torch_output = model(input_tensor)

# TTNN execution
device = ttnn.open_device(device_id=0)
input_reshaped = input_tensor.permute(0, 2, 3, 1).contiguous().reshape(1, 1, 1024000, 3)
ttnn_input = ttnn.from_torch(
    input_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

conv2d_config = ttnn.Conv2dConfig(
    weights_dtype=ttnn.bfloat16,
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    deallocate_activation=False,
    act_block_h_override=1024,
    enable_kernel_stride_folding=False,
    config_tensors_in_dram=True,
)
conv2d_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth, num_slices=0)

ttnn_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)
ttnn_bias = ttnn.from_torch(bias.reshape(1, 1, 1, 64), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)

common = dict(
    in_channels=3, out_channels=64, batch_size=1, input_height=800, input_width=1280, kernel_size=[7, 7],
    stride=[2, 2], padding=[3, 3, 3, 3], dilation=[1, 1], groups=1, device=device, conv_config=conv2d_config,
    slice_config=conv2d_slice_config,
)
ttnn_weight_prepared = ttnn.prepare_conv_weights(
    weight_tensor=ttnn_weight, input_memory_config=ttnn.DRAM_MEMORY_CONFIG, input_layout=ttnn.TILE_LAYOUT,
    weights_format="OIHW", has_bias=True, input_dtype=ttnn.bfloat16, output_dtype=ttnn.bfloat16, **common
)
ttnn_bias_prepared = ttnn.prepare_conv_bias(
    bias_tensor=ttnn_bias, input_memory_config=ttnn.DRAM_MEMORY_CONFIG, input_layout=ttnn.TILE_LAYOUT,
    input_dtype=ttnn.bfloat16, output_dtype=ttnn.bfloat16, **common
)


def pcc(x, y):
    x, y = x.to(torch.float64).flatten(), y.to(torch.float64).flatten()
    x, y = torch.nan_to_num(x, nan=0.0, posinf=1e30, neginf=-1e30), y
    vx, vy = x - x.mean(), y - y.mean()
    return float((vx @ vy) / (vx.norm() * vy.norm()))


def run(w, b, name):
    out = ttnn.conv2d(
        input_tensor=ttnn_input, weight_tensor=w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16, **common
    )
    got = ttnn.to_torch(ttnn.permute(ttnn.reshape(out, (1, 400, 640, 64)), (0, 3, 1, 2)))
    print(
        f"{name}: has inf {bool(torch.isinf(got).any())}, min/max {float(got.min()):.4g}/{float(got.max()):.4g}, "
        f"PCC vs torch {pcc(got, torch_output):.6f}",
        flush=True,
    )


run(ttnn_weight_prepared, ttnn_bias_prepared, "WITH prepare functions")
run(ttnn_weight, ttnn_bias, "WITHOUT prepare functions")
ttnn.close_device(device)
