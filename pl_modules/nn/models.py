import torch

import torch
from torch.nn import Module, ModuleList, Sequential, Linear, BatchNorm1d
from torch.nn.functional import silu, leaky_relu


class DenseNetwork(Module):
    def __init__(self, neurons_in, hidden_neurons, neurons_out, n_layers, activation_fn, *args, **kwargs):
        super(DenseNetwork, self).__init__()
        self.neurons_in = neurons_in
        self.hidden_neurons = hidden_neurons
        self.neurons_out = neurons_out
        self.n_layers = n_layers
        self.activation_fn = activation_fn

        self.initial_layer = Linear(neurons_in, hidden_neurons)
        self.hidden_layers = ModuleList([Linear(hidden_neurons, hidden_neurons) for _ in range(n_layers)])
        self.final_layer = Linear(hidden_neurons, neurons_out)

    def forward(self, x):
        h = self.activation_fn(self.initial_layer(x))
        for i in range(len(self.hidden_layers)):
            h = self.activation_fn(self.hidden_layers[i](h))
        out = self.final_layer(h)
        return out

