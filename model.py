from data import *

import torch
import pytorch_lightning as pl

from torch_geometric.nn import global_mean_pool
from torch.optim import Adam
from torch.nn.functional import relu
from torch.nn import Module, MultiheadAttention, Linear
from torch_geometric.nn import global_mean_pool, GATConv
from torch.optim.lr_scheduler import StepLR



class AtteMtion(Module):
    def __init__(self, in_channels, out_channels, heads):
        super(AtteMtion, self).__init__()
        self.lin_k = Linear(in_channels, out_channels)
        self.lin_q = Linear(in_channels, out_channels)
        self.lin_v = Linear(in_channels, out_channels)
        self.att = MultiheadAttention(out_channels, heads, batch_first=True)
    
    def forward(self, h):
        K = self.lin_k(h)
        Q = self.lin_q(h)
        V = self.lin_v(h)
        out = self.att(K[:, None, :], Q[:, None, :], V[:, None, :])
        return out


class InContextGNN(pl.LightningModule):
    def __init__(self):
        super(InContextGNN, self).__init__()
        self.graph1 = GATConv(in_channels=160, out_channels=16, heads=2)
        self.graph2 = GATConv(in_channels=32, out_channels=8, heads=8)
        self.att1 = AtteMtion(64, 8, 2)
        self.readout = Linear(8, 1)
        self.act = relu
        self.train_loader, self.val_loader = data("Mo", 10) 
 
    def forward(self, batch):
        graph_h1 = self.graph1(batch.x, batch.edge_index)
        graph_h1 = self.act(graph_h1)
        graph_h2 = self.graph2(graph_h1, batch.edge_index)
        graph_h2 = self.act(graph_h2)
        graph_h = global_mean_pool(graph_h2, batch.batch)
        graph_h = self.act(graph_h)
        h1 = self.att1(graph_h)
        h1 = self.act(h1[0])
        out = self.readout(h1[0:])
        return out

    def train_dataloader(self):
        train_loader = self.train_loader
        return train_loader

    def val_dataloader(self):
        val_loader = self.val_loader
        return val_loader

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.view(-1, 1))
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
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.view(-1, 1))
        self.log('val_loss', loss)
        return {'val_loss': loss}
