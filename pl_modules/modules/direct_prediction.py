import torch
import pytorch_lightning as pl
import hydra

import torch.nn.functional as F
from torch.optim import Adam

class GraphFeaturePredictor(pl.LightningModule):
    def __init__(self, gnn, readout=None, pool=None):
        super(GraphFeaturePredictor, self).__init__()
        self.gnn = hydra.utils.instantiate(gnn)
        self.pool = pool
        self.readout = hydra.utils.instantiate(readout)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        #scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

        return {
            'optimizer': optimizer}

    def forward(self, batch):
        node_representation = self.gnn(batch)
        if self.pool:
            graph_representation = self.pool(node_representation, batch.batch)
        else:
            graph_representation = node_representation
        # passing the entire batch to the decoder - perhaps coords/to_images should be used in a good encoder
        if self.readout:
            out = self.readout(graph_representation)
        else:
            out = graph_representation
        return out

    def general_step(self, batch, step_name):
        out = self(batch)
        loss = F.mse_loss(out, batch.y)
        self.log_dict({
            'train_loss': loss,
            }, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, 'test')
        return loss