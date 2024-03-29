import torch
import pytorch_lightning as pl
import hydra
import torch_scatter
from torch.optim.lr_scheduler import StepLR

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
    def __init__(self, encoder, decoder, label_readout, graph_pooling_fn=None, tricks=None, weight_loaders=None,
                 optimizer_cfg=None):  # pass cfg.model as argument to this
        super(InContextWrap, self).__init__()
        self.encoder = hydra.utils.instantiate(encoder, _recursive_=True)
        self.decoder = hydra.utils.instantiate(decoder, _recursive_=True)
        self.graph_pooling_fn = graph_pooling_fn
        self.label_readout = hydra.utils.instantiate(label_readout)
        self.optimizer_cfg = hydra.utils.instantiate(optimizer_cfg)

        if tricks:
            self.tricks = [hydra.utils.instantiate(t, _recursive_=True) for t in tricks]
        else:
            self.tricks = []

        if weight_loaders is None:
            weight_loaders = []
        self.weight_loaders = weight_loaders
        for loader in self.weight_loaders:
            loader.apply(self)


    @staticmethod
    def map_graph_outputs(graph_h, batch):
        num_rows = torch.max(batch.context) + 1
        num_cols = torch.max(batch.num_in_context) + 1
        num_channels = graph_h.shape[-1]


        x_index = batch.context
        y_index = batch.num_in_context

        flat_indices = x_index * num_cols + y_index
        flat_graphs = graph_h.reshape(-1, num_channels)

        flat_graphs = flat_graphs[y_index >= 0]
        flat_indices = flat_indices[y_index >= 0]


        out_tensor = torch_scatter.scatter_add(
            src=flat_graphs,
            index=flat_indices.reshape(-1, 1),
            dim_size=num_cols*num_rows,
            dim=0
            ).reshape(num_rows, num_cols, num_channels)
        return out_tensor

    @staticmethod
    def loss_at_labels(llm_out, y):
        y_predictions = llm_out[:, ::2]
        return torch.nn.functional.mse_loss(y_predictions, y)

    def forward(self, batch):
        if self.encoder:
            h = self.encoder(batch)
        else:
            h = batch.x
        if self.graph_pooling_fn:
            graph_h = self.graph_pooling_fn(h, batch.batch)
        else:
            graph_h = h

        graph_x = self.map_graph_outputs(graph_h, batch)
        graph_y = self.map_graph_outputs(batch.y.reshape(-1, 1), batch)
        # graph_x_old = graph_h.reshape(datapoints, graphs_per_datapoint, -1)
        # assert torch.all(torch.isclose(graph_x, graph_x_old))


        zs = self._combine(graph_x, graph_y)[:, :-1]
        llm_outs = self.decoder(inputs_embeds=zs).last_hidden_state
        outs = self.label_readout(llm_outs[:, ::2, :])
        last_out = outs[:, -1, :]
        last_y = graph_y[:, -1, :]
        ys = graph_y

        return last_out, last_y, outs, ys

    @staticmethod
    def _combine(xs_b, ys_b):
        """Integrates the x's and the y's into a single sequence."""
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
                batch = trick.apply(batch, split=step)
        return batch

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        scheduler = {'scheduler': StepLR(optimizer, self.optimizer_cfg.step_size,
                     gamma=self.optimizer_cfg.gamma), 'interval': 'epoch'}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch = self.apply_tricks(batch, step='train')
        graphs_per_datapoint = torch.max(batch.num_in_context) + 1
        out, label, outs, labels = self(batch)
        single_loss = torch.nn.functional.mse_loss(out, label)
        full_loss = torch.nn.functional.mse_loss(outs, labels)
        loss = full_loss
        self.log_dict({'train_loss': loss,
                       'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                       }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.apply_tricks(batch, step='val')
        graphs_per_datapoint = torch.max(batch.num_in_context) + 1
        out, label, outs, labels = self(batch)
        single_loss = torch.nn.functional.mse_loss(out, label)
        full_loss = torch.nn.functional.mse_loss(outs, labels)
        loss = full_loss
        self.log_dict({'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch = self.apply_tricks(batch, step='test')
        graphs_per_datapoint = torch.max(batch.num_in_context) + 1
        out, label, outs, labels = self(batch)
        loss = torch.nn.functional.mse_loss(out, batch.y.reshape(torch.max(batch.context) + 1, graphs_per_datapoint)[:, [-1]])
        self.log_dict({'test_loss': loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
