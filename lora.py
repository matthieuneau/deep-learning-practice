# for now, only handle 3x3 conv with padding=stride=1
import torch.nn as nn


class ConvLora(nn.Module):
    def __init__(self, in_channels, out_channels, frozen_weights, r=2):
        super().__init__()
        self.downConv = nn.Conv2d(in_channels, r, kernel_size=3, padding=1, stride=1)
        self.upConv = nn.Conv2d(r, out_channels, kernel_size=1, padding=0, stride=1)
        self.frozen_weights = frozen_weights
        
    def float(self, x):
        main_x = self.frozen_weights(x) 
        lora_x = self.downConv(x)
        lora_x = self.upConv(lora_x)
        return main_x + lora_x