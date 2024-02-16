from typing import Optional, Callable, List

import torch

from torch.nn import ReLU
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch.nn import Module
from torch_geometric.nn import GAT, GCN
from pl_modules.imports.mxmnet.model import MXMNet
from pl_modules.imports.mxmnet.model import Config as MXMNetConfig

from torch_geometric.nn.models.basic_gnn import BasicGNN


class GATWrap(Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, **kwargs):
        super(GATWrap, self).__init__()
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=out_channels,
                         **kwargs)

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index)


class GATv2(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        if 'concat' in kwargs:
            del kwargs['concat']

        if 'heads' in kwargs:
            assert hidden_channels % kwargs['heads'] == 0

        out_channels = hidden_channels // kwargs.get('heads', 1)

        self.convs.append(
            GATv2Conv(in_channels, out_channels, dropout=dropout, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(GATv2Conv(hidden_channels, out_channels, **kwargs))


class GATv2Wrap(Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, heads=3, **kwargs):
        super(GATv2Wrap, self).__init__()
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=out_channels,
                         heads=heads,
                         **kwargs)

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index)


class GCNWrap(Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, **kwargs):
        super(GCNWrap, self).__init__()
        self.model = GCN(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=out_channels,
                         **kwargs)

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index)


class MXMNetWrap(Module):
    def __init__(self, mxmnet_config: MXMNetConfig):
        super(MXMNetWrap, self).__init__()
        self.model = MXMNet(mxmnet_config)

    def forward(self, batch):
        return self.model(batch)


class DimeNetWrap(Module):
    def __len__(self, dimenet_model):
        super(DimeNetWrap, self).__init__()
        self.model = dimenet_model

    def forward(self, batch):
        return self.model(batch)


if __name__ == "__main__":
    model = GATv2Wrap(
        in_channels=58,
        hidden_channels=32,
        num_layers=3,
        out_channels=64,
        heads=3
    )