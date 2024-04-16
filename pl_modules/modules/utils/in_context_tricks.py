import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from abc import ABC, abstractmethod


class Curriculum:
    def __init__(self, start, end, increment, interval):
        self.start = start
        self.end = end
        self.increment = increment
        self.interval = interval
        self.value = start
        self.current_step = 0
    def step(self):
        self.current_step += 1
        if self.value >= self.end:
            pass
        else:
            if self.current_step % self.interval == 0:
                self.value += self.increment
                print(f'Curriculum new value {self.value}')
        return self.value


class Constant:
    def __init__(self, value):
        self.value = value

    def step(self):
        return self.value


class Trick(ABC):
    def __init__(self, *args, **kwargs):
        self.step_counter = 0
        self.curriculum = None


    @abstractmethod
    def apply(self, batch: torch_geometric.data.Batch, split) -> torch_geometric.data.Batch:
        pass

class ShuffleTrick(Trick, ABC):
    def __init__(self):
        super(ShuffleTrick, self).__init__()

    def apply(self, batch, split):
        n_cols = torch.max(batch.num_in_context) + 1
        n_rows = torch.max(batch.context) + 1

        # Apply permutations to the tensor to shuffle each row independently
        # We use torch.arange(n_rows)[:, None] to ensure correct broadcasting over rows
        context_map = batch.num_in_context.reshape(n_rows, n_cols)
        perm = torch.stack([torch.randperm(n_cols, device=context_map.device) for _ in range(n_rows)])
        batch.num_in_context = context_map[torch.arange(n_rows)[:, None], perm].reshape(-1)

        #print("Shuffled Tensor:\n", batch.num_in_context)
        return batch


class MaskX(Trick, ABC):
    def __init__(self, curriculum):
        super(MaskX, self).__init__()
        self.curriculum = curriculum


    def apply(self, batch, split):
        n_channels = batch.x.shape[1]
        n_values = self.curriculum.step()
        mask = torch.tensor([1.] * n_values + [0] * (n_channels-n_values)).reshape(1, -1)
        batch.x = batch.x * mask
        return batch


class NoiseY(Trick, ABC):
    def __init__(self, mu_curriculum, sigma_curriculum):
        self.mu_curriculum = mu_curriculum
        self.sigma_curriculum = sigma_curriculum


    def apply(self, batch, split):
        n_contexts = batch.x.shape[0]

        mu = self.mu_curriculum.step()
        sigma = self.sigma_curriculum.step()

        noises = (torch.randn(n_contexts)+mu)*sigma.reshape(-1, 1)
        batch.y = batch.y + noises
        return batch



class MaskContext(Trick, ABC):
    def __init__(self, curriculum):
        super(MaskContext, self).__init__()
        self.curriculum = curriculum
    def apply(self, batch, split):
        max_examples = self.curriculum.step()
        batch.num_in_context = torch.where(batch.num_in_context < max_examples, batch.num_in_context, torch.tensor(-1))
        return batch