# for now, only handle 3x3 conv with padding=stride=1
import torch.nn as nn


class ConvLora(nn.Module):
    def __init__(self, frozen_conv, r=2, alpha=2):
        super().__init__()
        self.alpha = alpha
        self.r = r
        self.downConv = nn.Conv2d(
            frozen_conv.in_channels,
            r,
            frozen_conv.kernel_size,
            frozen_conv.stride,
            frozen_conv.padding,
        )
        self.upConv = nn.Conv2d(
            r, frozen_conv.out_channels, kernel_size=1, padding=0, stride=1
        )
        self.frozen_conv = frozen_conv

    def forward(self, x):
        main_x = self.frozen_conv(x)
        lora_x = self.downConv(x)
        lora_x = self.upConv(lora_x)
        return main_x + (self.alpha / self.r) * lora_x
