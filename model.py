import numpy as np
import os
import json
from pathlib import Path
import re
from time import sleep
from tqdm import tqdm
import warnings

import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split



from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure, Lattice, Site

import torch
import pytorch_lightning as pl

from torch_geometric.nn import global_mean_pool
from torch.optim import Adam
from torch.nn.functional import relu
from torch.nn import Module, MultiheadAttention, Linear
from torch_geometric.nn import global_mean_pool, GATConv
from torch.optim.lr_scheduler import StepLR

from transformers import GPT2Config, GPT2Model

from data import *


class GPT2BasedModel(Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(GPT2BasedModel, self).__init__()

        # GPT-2 Configuration
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self._read_in = Linear(n_dims, n_embd)
        self._read_out = Linear(n_embd, 1)
        self._read_in_y = Linear(1, 1)

        self._backbone = GPT2Model(configuration)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, input_tensor, label_tensor):
        y = self._read_in_y(label_tensor)
        x = self._read_in(input_tensor)
        zs = self._combine(x, y)[:, :-1]
        gpt2_output = self._backbone(inputs_embeds=zs)
        output = self._read_out(gpt2_output.last_hidden_state[:, -1, :])

        return output


class InContextGNN(pl.LightningModule):
    def __init__(self):
        super(InContextGNN, self).__init__()
        self.graph1 = GATConv(in_channels=160, out_channels=16, heads=2)
        self.graph2 = GATConv(in_channels=32, out_channels=8, heads=8)
        self.att1 = GPT2BasedModel(64, 128)
        self.readout = Linear(1, 4)
        self.act = relu
        self.train_loader, self.val_loader = data("Mo", 4, 4) 
 
    def forward(self, batch):
        #encoder
        graphs_per_datapoint = torch.max(batch.config_label) + 1
        actual_batch_dot_batch = batch.batch * graphs_per_datapoint + batch.config_label
	
        graph_h1 = self.graph1(batch.x, batch.edge_index)
        graph_h1 = self.act(graph_h1)
        graph_h2 = self.graph2(graph_h1, batch.edge_index)
        graph_h2 = self.act(graph_h2)
        graph_h = global_mean_pool(graph_h2, actual_batch_dot_batch)
        
        o = batch.y.reshape(-1, 1)
        graph_h = graph_h.reshape(torch.max(batch.batch) + 1 , graphs_per_datapoint, -1)

        h1 = self.att1(graph_h, o)
        h1 = self.act(h1)
        out = self.readout(h1)
        return out

    def train_dataloader(self):
        train_loader = self.train_loader
        return train_loader

    def val_dataloader(self):
        val_loader = self.val_loader
        return val_loader

    def training_step(self, batch, batch_idx):
        graphs_per_datapoint = torch.max(batch.config_label) + 1
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint)[:, -1])
        self.log('train_loss', loss)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', 
                'monitor': 'val_loss',
            }
        }

    def validation_step(self, batch, batch_idx):
        graphs_per_datapoint = torch.max(batch.config_label) + 1
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint)[:, -1])
        self.log('val_loss', loss)
        return {'val_loss': loss}
