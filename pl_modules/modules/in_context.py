import torch
import pytorch_lightning as pl
import hydra

from torch.optim import Adam

class InContextWrap(pl.LightningModule):
    """ This class is designed to work just like InContextGNN provided correct configuration.
    Layers will all be encapsulated inside encoder/decoder submodules. Encoder can be seen as a submodule
    that starts with a node-level representation and ends up with graph-level representation. Decoder
    takes those representations to yield a final estimate. As _combine is symbolic and quite peculiar,
    it is left inside this wrapper class.
    To sum up, this class will not be equipped with optimizer/dataloaders. I will make alternative train.py
    and if you like that, we can merge to a single approach.
    """
    def __init__(self, encoder, decoder, label_readout, graph_pooling_fn=None, tricks=None): # pass cfg.model as argument to this
        super(InContextWrap, self).__init__()
        self.encoder = hydra.utils.instantiate(encoder, _recursive_=True)
        self.decoder = hydra.utils.instantiate(decoder, _recursive_=True)
        self.graph_pooling_fn = graph_pooling_fn
        self.label_readout = hydra.utils.instantiate(label_readout)
        self.tricks = tricks
        # going to wrap all extra nn in the decoder, so they fall off
        # no datasets here, I will write an example datamodule for my model
        # this model will be designed to behave just like InContextGNN, but with enhanced modularity

    def forward(self, batch):
        # encoder
        graphs_per_datapoint = torch.max(batch.context_num) + 1
        actual_batch_dot_batch = batch.batch * graphs_per_datapoint + batch.context_num
        # passing the entire batch to the decoder - perhaps coords/to_images should be used in a good encoder
        h = self.encoder(batch)
        if self.graph_pooling_fn:
            graph_h = self.graph_pooling_fn(h, actual_batch_dot_batch)
        graph_x = graph_h.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint, -1)

        zs = self._combine(graph_x, batch.y)[:, :-1]

        llm_out = self.decoder(inputs_embeds=zs).last_hidden_state[:, -1, :]

        out = self.label_readout(llm_out)
        return out

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

    def apply_tricks(self, batch, step='train'):
        if self.tricks is not None:
            for trick in self.tricks:
                batch = trick.apply(batch, step=step)
        return batch

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)

        return {
            'optimizer': optimizer,
        }

    def training_step(self, batch, batch_idx):
        batch = self.apply_tricks(batch, step='train')
        graphs_per_datapoint = torch.max(batch.context_num) + 1
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint)[:,-1])
        self.log_dict({'train_loss': loss,
                       'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                       }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.apply_tricks(batch, step='val')
        graphs_per_datapoint = torch.max(batch.context_num) + 1
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch.y.reshape(torch.max(batch.batch) + 1, graphs_per_datapoint)[:,[-1]])
        self.log_dict({'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
