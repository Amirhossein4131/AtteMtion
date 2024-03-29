import torch
import pytorch_lightning as pl
from torch import nn
import hydra
from torch_sparse import SparseTensor
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn.conv import TransformerConv
from pl_modules.imports.dimenet.model import radius_graph_pbc_wrapper, BesselBasisLayer, SphericalBasisLayer
from pl_modules.imports.dimenet.utils import frac_to_cart_coords, get_pbc_distances

from torch.optim import Adam

import math
from torch import Tensor


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class InContextTransformer(pl.LightningModule):
    def __init__(self, node_channels=3, n_graph_channels=64, max_positions=128, tricks=None, weight_loaders=None,
                 optimizer_cfg=None, otf_graph=True):  # pass cfg.model as argument to this
        super(InContextTransformer, self).__init__()
        self.otf_graph = otf_graph
        self.cutoff = 10.0
        self.max_num_neighbors = 50
        cutoff = self.cutoff
        num_radial = 6
        envelope_exponent = 5
        num_spherical = 7
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.optimizer_cfg = optimizer_cfg
        self.n_graph_channels = n_graph_channels
        self.positional_embedding = nn.Embedding(max_positions, n_graph_channels)
        self.blocks = nn.ModuleList([
            InContextBlock(node_channels=node_channels, graph_channels=n_graph_channels) for _ in range(12)])

        self.injection = InjectionBlock(n_graph_channels)
        self.readout = ReadOutBlock(n_graph_channels)

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
    def loss_at_labels(llm_out, y):
        y_predictions = llm_out[:, ::2]
        return torch.nn.functional.mse_loss(y_predictions, y)

    def _otf_graph(self, data):
        edge_index, cell_offsets, neighbors = radius_graph_pbc_wrapper(
            data, self.cutoff, self.max_num_neighbors, data.num_atoms.device
        )
        data.edge_index = edge_index
        data.to_jimages = cell_offsets
        data.num_bonds = neighbors
        return data

    def get_basis(self, data):
        pos = frac_to_cart_coords(
            data.frac_coords,
            data.lengths,
            data.angles,
            data.num_atoms)

        out = get_pbc_distances(
            data.frac_coords,
            data.edge_index,
            data.lengths,
            data.angles,
            data.to_jimages,
            data.num_atoms,
            data.num_bonds,
            return_offsets=True
        )

        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]

        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=data.atom_types.size(0)
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i + offsets[idx_ji],
            pos[idx_k].detach() - pos_j + offsets[idx_kj],
        )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        return rbf, sbf

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        # row, col = col, row  # Swap because my definition of edge_index is i->j

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, batch):
        if self.otf_graph:
            batch = self._otf_graph(batch)

        rbf, sbf = self.get_basis(batch)

        # initialize graph features as zeros
        n_graphs = batch.context.shape[0]   # number of graphs
        n_graphs_reps = 2 * n_graphs        # number of graphs plus number of labels
        batch.graph_h = torch.zeros((n_graphs_reps, self.n_graph_channels), dtype=batch.x.dtype, device=batch.x.device)

        # add positional embeddings
        full_num_in_context = torch.stack([batch.num_in_context*2, (batch.num_in_context*2)+1], dim=1).reshape(-1)
        positional_emb = self.positional_embedding(full_num_in_context)
        batch.graph_h + positional_emb

        # get context ptr
        temp_placeholder = torch.tensor([-1], dtype=batch.context.dtype, device=batch.context.device)
        temp = torch.concatenate([temp_placeholder, batch.context, temp_placeholder])
        batch.context_ptr = torch.where(torch.not_equal(temp[1:], temp[:-1]))[0]

        # create context edge index
        context_edge_index_list = []
        for i, j in zip(batch.context_ptr[:-1], batch.context_ptr[1:]):
            full_adj = torch.ones(j-i, j-i, dtype=batch.edge_index.dtype, device=batch.ptr.device)
            masked_adj = torch.tril(full_adj)
            tgt, src = torch.where(masked_adj)
            partial_context_edge_index = torch.stack([src, tgt], dim=0)
            partial_context_edge_index = partial_context_edge_index + i
            context_edge_index_list.append(partial_context_edge_index)
        batch.context_edge_index = torch.concatenate(context_edge_index_list, dim=1)

        # create upward/downward edge index
        down = torch.arange(batch.x.shape[0])
        up = batch.batch * 2
        batch.upward = torch.stack([down, up], dim=0)
        batch.downward = torch.stack([up, down], dim=0)

        batch.h = batch.x
        batch = self.injection(batch)

        for block in self.blocks:
            batch = block(batch)

        outs = self.readout(batch)
        last_out = outs[(batch.context_ptr - 1)[1:]]
        ys = batch.y
        last_y = ys[(batch.context_ptr - 1)[1:]]

        return last_out, last_y, outs, ys

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


class InContextBlock(nn.Module):
    def __init__(self, node_channels, graph_channels):
        super(InContextBlock, self).__init__()
        self.node_interaction = TransformerConv(node_channels, node_channels)
        self.upward_pass = TransformerConv((node_channels, graph_channels), graph_channels)
        self.graph_interaction = TransformerConv(graph_channels, graph_channels)
        self.downward_pass = TransformerConv((graph_channels, node_channels), node_channels)

    def forward(self, batch):
        batch.h = self.node_interaction(batch.h, batch.edge_index)
        batch.graph_h = self.upward_pass((batch.h, batch.graph_h), batch.upward)
        batch.graph_h = self.graph_interaction(batch.graph_h, batch.context_edge_index)
        batch.h = self.downward_pass((batch.graph_h, batch.h), batch.downward)
        # Linear, Activation layers, norms

        return batch


class InjectionBlock(nn.Module):
    def __init__(self, graph_channels, n_labels=1):
        super(InjectionBlock, self).__init__()
        self.linear = nn.Linear(n_labels, graph_channels)

    def forward(self, batch):
        encoded_y = self.linear(batch.y)
        encoded_y[(batch.context_ptr - 1)[1:]] *= 0
        batch.graph_h[1::2] += encoded_y
        return batch


class ReadOutBlock(nn.Module):
    def __init__(self, graph_channels, n_labels=1):
        super(ReadOutBlock, self).__init__()
        self.linear = nn.Linear(graph_channels, n_labels)

    def forward(self, batch):
        outputs = self.linear(batch.graph_h[::2])
        return outputs


activation_dict = {'gelu': NewGELUActivation,
                   'sigmoid': nn.Sigmoid,
                   'silu': nn.SiLU,
                   'relu': nn.ReLU}


class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, activation_fn, output_channels):
        super(MLP, self).__init__()
        if output_channels is None:
            output_channels = input_channels

        self.linear_1 = nn.Linear(input_channels, hidden_channels)
        self.linear_2 = nn.Linear(hidden_channels, output_channels)
        if activation_fn:
            self.act = activation_dict[activation_fn]()

    def forward(self, x):
        h1 = self.linear_1(x)
        if self.activation_fn:
            ha = self.act(h1)
        else:
            ha = h1

        h2 = self.linear_2(ha)

        return h2







