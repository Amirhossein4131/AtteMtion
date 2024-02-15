import torch
import torch_geometric
import pytorch_lightning as pl

from torch.nn import Module
from torch_geometric.nn import GATv2Conv, GAT, GCN
from pl_modules.imports.mxmnet.model import MXMNet
from pl_modules.imports.mxmnet.model import Config as MXMNetConfig


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


class GATv2ConvWrap(Module):
    def __init__(self, in_channels, heads=3, out_channels=None, **kwargs):
        super(GATv2ConvWrap, self).__init__()
        self.model = GATv2Conv(in_channels=in_channels,
                               heads=heads,
                               out_channels=out_channels,
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
