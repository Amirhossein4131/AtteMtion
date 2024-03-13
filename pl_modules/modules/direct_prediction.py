import torch
import pytorch_lightning as pl
import hydra
from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F
from torch.optim import Adam


class GraphFeaturePredictor(pl.LightningModule):
    def __init__(self, gnn, readout=None, pool=None, optimizer_cfg=None):
        super(GraphFeaturePredictor, self).__init__()
        self.gnn = hydra.utils.instantiate(gnn)
        self.pool = pool
        self.readout = hydra.utils.instantiate(readout)
        self.optimizer_cfg = hydra.utils.instantiate(optimizer_cfg)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        scheduler = {'scheduler': StepLR(optimizer, self.optimizer_cfg.step_size,
                     gamma=self.optimizer_cfg.gamma), 'interval': 'epoch'}
        return [optimizer], [scheduler]

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
        # loss_check = F.mse_loss(torch.zeros_like(out), batch.y)
        # loss_check2 = F.mse_loss(torch.full_like(out, mean_label), batch.y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, step_name='train')
        self.log_dict({
            'train_loss': loss, 'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
            }, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, step_name='val')
        self.log_dict({
            'val_loss': loss,
            }, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        pass