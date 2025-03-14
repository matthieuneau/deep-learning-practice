import torch
import torch.nn as nn

from lora import merge_lora_conv


def merge_lora_conv(
    downconv: nn.Conv2d,
    upconv: nn.Conv2d,
    Cin: int,
    Cout: int,
    r: int,
    kernel_size: int,
) -> nn.Conv2d:
    """
    Merge two convolutions into a single convolution. The second
    Args:
        downconv: The first convolution (r x Cin x kernel_size x kernel_size)
        upconv: The second convolution (Cout x r x 1 x 1)
        Cin: Number of input channels
        Cout: Number of output channels
        r: Rank of the factorized convolution
        kernel_size: Size of the convolution kernel
    """
    # Two-factor convolution (factorized LoRA-style)
    downconv = nn.Conv2d(Cin, r, kernel_size=kernel_size, padding=1, bias=True)
    upconv = nn.Conv2d(r, Cout, kernel_size=1, bias=True)

    # The merged convolution
    merged_conv = nn.Conv2d(Cin, Cout, kernel_size=kernel_size, padding=1, bias=True)

    with torch.no_grad():
        # Merge weights via einsum (conv2 is 1x1, so weight[:, :, 0, 0] is our channel-mixing matrix)
        # shape of conv1.weight: (r, Cin, kH, kW)
        # shape of conv2.weight: (Cout, r, 1, 1) -> can be viewed as (Cout, r)
        merged_conv.weight.copy_(
            torch.einsum("or,rchw->ochw", upconv.weight[:, :, 0, 0], downconv.weight)
        )
        # Merge bias
        # conv2.bias (Cout) + conv2.weight (Cout x r) @ conv1.bias (r)
        merged_conv.bias.copy_(
            upconv.bias + torch.matmul(upconv.weight[:, :, 0, 0], downconv.bias)
        )

    # Test with a small random input
    x = torch.randn(1, Cin, 4, 4)
    y_factorized = upconv(downconv(x))
    y_merged = merged_conv(x)

    print("Outputs close? ", torch.allclose(y_factorized, y_merged, atol=1e-6))
    return merged_conv


if __name__ == "__main__":
    Cin = 64
    Cout = 512
    r = 2
    kernel_size = 3
    merge_lora_conv(nn.Conv2d, nn.Conv2d, Cin, Cout, r, kernel_size)
