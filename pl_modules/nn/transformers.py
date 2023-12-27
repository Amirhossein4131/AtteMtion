import torch
import torch_geometric
import pytorch_lightning as pl
import transformers

from torch.nn import Module, Linear
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import MessagePassing

from transformers import GPT2Model, GPT2Config


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
        output = gpt2_output.last_hidden_state[:, -1, :]

        return output

