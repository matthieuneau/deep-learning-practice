import torch.nn as nn
from torch_geometric.nn import GAT


class StudentModel(GAT):
    def __init__(self):
        super().__init__(
            in_channels=50,
            hidden_channels=256,
            num_layers=4,
            out_channels=121,
            act="ELU",
            heads=4,
        )

        self.skips = nn.ModuleList()
        for conv in self.convs:
            effective_out = (
                conv.out_channels
                if hasattr(conv, "concat") and not conv.concat
                else conv.out_channels * conv.heads
            )
            if conv.in_channels != effective_out:
                self.skips.append(nn.Linear(conv.in_channels, effective_out))
            else:
                self.skips.append(nn.Identity())

    def forward(self, x, edge_index):
        for conv, skip in zip(self.convs, self.skips):
            residual = skip(x)
            x = conv(x, edge_index)
            x += residual
            x = self.act(x)
        return x
