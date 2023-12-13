import torch
import pytorch_lightning as pl
import hydra

class InContextWrap(pl.LightningModule):
    """ This class is designed to work just like InContextGNN provided correct configuration.
    Layers will all be encapsulated inside encoder/decoder submodules. Encoder can be seen as a submodule
    that starts with a node-level representation and ends up with graph-level representation. Decoder
    takes those representations to yield a final estimate. As _combine is symbolic and quite peculiar,
    it is left inside this wrapper class.
    To sum up, this class will not be equipped with optimizer/dataloaders. I will make alternative train.py
    and if you like that, we can merge to a single approach.
    """
    def __init__(self, cfg): # pass cfg.model as argument to this
        super(InContextWrap, self).__init__()
        self.encoder = hydra.utils.instantiate(cfg.encoder)
        self.decoder = hydra.utils.instantiate(cfg.decoder)
        # going to wrap all extra layers in the decoder, so they fall off
        # no datasets here, I will write an example datamodule for my model
        # this model will be designed to behave just like InContextGNN, but with enhanced modularity

    def forward(self, batch):
        # encoder
        graphs_per_datapoint = torch.max(batch.config_label) + 1
        actual_batch_dot_batch = batch.batch * graphs_per_datapoint + batch.config_label
        # passing the entire batch to the decoder - perhaps coords/to_images should be used in a good encoder
        graph_h = self.encoder(batch)
        graph_x = graph_h.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint, -1)

        out = self.decoder(graph_x, batch.y)
        return out

    def training_step(self, batch, batch_idx):
        graphs_per_datapoint = torch.max(batch.config_label) + 1
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint)[:,-1])
        self.log_dict({'train_loss': loss,
                       'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                       }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        graphs_per_datapoint = torch.max(batch.config_label) + 1
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint)[:, [-1]])
        self.log_dict({'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
