import torch
from torch.nn import Module, Linear
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

    def forward(self, input_tensor, label_tensor):
        y = self._read_in_y(label_tensor)
        x = self._read_in(input_tensor)
        zs = self._combine(x, y)[:, :-1]
        gpt2_output = self._backbone(inputs_embeds=zs)
        output = gpt2_output.last_hidden_state[:, -1, :]

        return output
