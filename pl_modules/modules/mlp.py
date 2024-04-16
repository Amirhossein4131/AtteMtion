import torch
from torch import nn


activation_dict = {'sigmoid': nn.Sigmoid,
                   'silu': nn.SiLU,
                   'relu': nn.ReLU}


class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels=None, output_channels=None, activation_fn="silu"):
        super(MLP, self).__init__()
        if output_channels is None:
            output_channels = input_channels

        if hidden_channels is None:
            hidden_channels = input_channels

        self.activation_fn = activation_fn
        self.linear_1 = nn.Linear(input_channels, hidden_channels)
        self.linear_2 = nn.Linear(hidden_channels, output_channels)
        if activation_fn:
            self.act = activation_dict[activation_fn]()

    def forward(self, batch):
        x = batch.strain

        h1 = self.linear_1(x)

        if self.activation_fn:
            ha = self.act(h1)
        else:
            ha = h1

        h2 = self.linear_2(ha)
        return h2
