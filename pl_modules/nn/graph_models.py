import torch
import torch_geometric
import pytorch_lightning as pl

from torch.nn import Module
from torch_geometric.nn import GATv2Conv, GAT, GCN

class GATWrap(Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, **kwargs):
        """
        Just a simple wrapper so that not as to abstract away the GAT network that's in torch_geometric.
        You could potentially import this directly, fully through hydra, but:
        a) it would be less intuitive if something from external package was loaded by hydra and never visible
        in our project
        b) the wrapper is a bit more convenient because of the consistent signature: we will be able
        to pass batch as an argument, instead of batch.x, batch.edge index, which this model normally would
        take
        """
        # todo it's an internal comment to be removed by the end of the project
        super(GATWrap, self).__init__()
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
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



