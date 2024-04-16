import sklearn.preprocessing
import torch
import pytorch_lightning as pl
import torch_geometric
import numpy as np
import random
from tqdm import tqdm

from pl_modules.data.utils.in_context_dataloader import InContextDataLoader
# from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from torch_geometric.data import Data, Batch
import hydra
from omegaconf import OmegaConf, DictConfig
from torch_geometric.utils import erdos_renyi_graph

import os
from torch_geometric.utils import erdos_renyi_graph
from torch.utils.data import Dataset
from typing import Union, Optional, List, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import hydra
import json
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data


@dataclass
class ContextGraphData:
    data: List[Data]
    indices: torch.Tensor


class ContextMode(Enum):
    CONTEXT = auto()
    SEPARATE = auto()
    NO_CONTEXT = auto()


class ContextGraphDataset(Dataset):
    def __init__(self, context_data: ContextGraphData):
        """
        Custom dataset initialization.
        :param indices: A tensor of indices representing the data.
        :param data_strings: A list of strings representing the data objects.
        """
        self.indices = context_data.indices
        self.data_objs = context_data.data

    def __len__(self):
        """
        Return the number of batches (rows in the indices tensor).
        """
        return self.indices.size(0)

    def __getitem__(self, idx):
        """
        Retrieve strings corresponding to all indices in the specified row(s) of the tensor.
        :param idx: Index (or slice) of the batch(es) (row number(s) in the indices tensor).
        :return: A list of strings corresponding to the indices.
        """
        try:
            row_indices = self.indices[idx].flatten()
            batch_data = [self.data_objs[i] for i in row_indices]
            return batch_data, row_indices
        except:
            print('problem')


class InContextDataModule(pl.LightningDataModule, ABC):
    def __init__(self, mode: Union[ContextMode, str], *args, **kwargs):
        super().__init__()
        self._set_mode(mode)
        self.train: Union[ContextGraphData, None] = None
        self.val: Union[ContextGraphData, None] = None
        self.test: Union[ContextGraphData, None] = None

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def _set_mode(self, mode: Union[ContextMode, str]):
        if isinstance(mode, str):
            self.mode = getattr(ContextMode, mode.upper())
        else:
            self.mode = mode

    @abstractmethod
    def context_dataloader(self, split: str = 'train'):
        raise NotImplementedError

    @abstractmethod
    def separate_dataloader(self, split: str = 'train'):
        raise NotImplementedError

    @abstractmethod
    def no_context_dataloader(self, split: str = 'train'):
        raise NotImplementedError

    def general_dataloader(self, split: str):
        method_mapping = {
            ContextMode.CONTEXT: self.context_dataloader,
            ContextMode.SEPARATE: self.separate_dataloader,
            ContextMode.NO_CONTEXT: self.no_context_dataloader,
        }
        if self.mode in method_mapping:
            return method_mapping[self.mode](split=split)
        else:
            raise ValueError(f"The mode {self.mode} value not recognized")

    def train_dataloader(self):
        return self.general_dataloader(split='train')

    def val_dataloader(self):
        return self.general_dataloader(split='val')

    def test_dataloader(self):
        return self.general_dataloader(split='test')

    @staticmethod
    def instantiate(node: Union[OmegaConf, DictConfig, Any]):
        if isinstance(node, (dict, OmegaConf, DictConfig)):
            if '_target_' in node.keys():
                return hydra.utils.instantiate(node)
        return node


class DimeNetDataModule(InContextDataModule, ABC):
    def __init__(self, train_csv, test_csv, elements, label_scaler=None,
                 batch_size=1, mode='context', separate_test=False, *args, **kwargs):
        super(DimeNetDataModule, self).__init__(mode=mode)
        self.batch_size = batch_size
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.elements = elements
        self.label_scaler = hydra.utils.instantiate(label_scaler)

        # Read atom types
        with open(elements) as file:
            self.elements = json.load(file)

        self.train_dataset = self.getdb(self, train_csv)
        self.train = ContextGraphData(self.train_dataset[:8000], torch.arange(8000).reshape(-1, 10))
        self.val = ContextGraphData(self.train_dataset[8000:9000], torch.arange(1000).reshape(-1, 10))
        self.test = ContextGraphData(self.train_dataset[9000:], torch.arange(1000).reshape(-1, 10))


        # if separate_test:
        #     self.test_dataset = self.getdb(self, test_csv)
        # else:
        #     self.train_size = int(0.8 * len(self.train_dataset))
        #     self.test_size = len(self.train_dataset) - self.train_size
        #     self.train_dataset, self.test_dataset = random_split(self.train_dataset,
        #                                                      [self.train_size, self.test_size])

        if self.label_scaler is not None:
            self.label_scaler.fit(torch.stack(
                            [t.y for t in self.train_dataset],).reshape(-1, 1).numpy())

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    @staticmethod
    def getdb(self, dataset_csv):
        data = pd.read_csv(dataset_csv, skipinitialspace=True)
        db = []
        for i in range(len(data)):
            cif = data['cif'][i]
            lines = cif.strip().split('\n')

            # target (y)
            prop = data['stress_Gpa'][i]
            strain = data['strain'][i]

            # fractional coordinates
            frac_cords = np.array([line.split()[3:-1] for line in lines[25:]], dtype=float)

            # atom types and number of atoms
            atom_types = [self.elements[line.split()[0]] for line in lines[25:]]
            num_atoms = len(atom_types)

            # Angles
            angles = np.array([line.split()[1] for line in lines[6:9]], dtype=float)

            # Lengths
            lengths = np.array([line.split()[1] for line in lines[3:6]], dtype=float)

            data_point = Data(x=torch.tensor(frac_cords, dtype=torch.float32),
                              frac_coords=torch.tensor(frac_cords, dtype=torch.float32),
                              atom_types=torch.tensor(atom_types, dtype=torch.int),
                              lengths=torch.tensor(lengths, dtype=torch.float32).view(1, -1),
                              angles=torch.tensor(angles, dtype=torch.float32).view(1, -1),
                              num_atoms=torch.tensor(num_atoms),
                              strain=torch.tensor(strain, dtype=torch.float32).view(1, -1),
                              y=torch.tensor(prop, dtype=torch.float32).view(1, -1))

            db.append(data_point)
        return db

    def scale_data2(self, dataset):
        yd = [d.y for d in dataset]
        yt = torch.cat(yd, dim=0)
        yt = torch.tensor(self.label_scaler.transform(yt), dtype=torch.float32)
        dataset = dataset.copy()
        for n, data in enumerate(dataset):
            dataset[n].y = yt[n:n+1]

        return dataset

    def scaled_data(self, data):
        return data

    def context_dataloader(self, split: str = 'train'):
        data = getattr(self, split)
        return InContextDataLoader(ContextGraphDataset(self.scaled_data(data)), shuffle=True, batch_size=self.batch_size)

    def separate_dataloader(self, split: str = 'train'):
        data = getattr(self, split)
        data.indices = data.indices.reshape(-1, 1)
        return InContextDataLoader(ContextGraphDataset(self.scaled_data(data)), shuffle=False, batch_size=self.batch_size)

    def no_context_dataloader(self, split: str = 'train'):
        data = getattr(self, split)
        data.indices = data.indices[:, -1]
        return InContextDataLoader(ContextGraphDataset(self.scaled_data(data)), shuffle=True, batch_size=self.batch_size)


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    train_path = os.path.join(os.environ['PROJECT_ROOT'], 'data', 'strain', 'db.csv')
    elements_path = os.path.join(os.environ['PROJECT_ROOT'], 'data', 'strain', 'elements.json')

    datamodule = DimeNetDataModule(train_csv=train_path, test_csv=train_path, elements=elements_path,
                                   separate_test=False, batch_size=16, mode='context')
    train = datamodule.train_dataloader()
    batch = next(iter(train))
    val = datamodule.val_dataloader()

