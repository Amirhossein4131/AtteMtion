import sklearn.preprocessing
import torch
import pytorch_lightning as pl
import torch_geometric
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from torch_geometric.data import Data
import hydra
from pl_modules.data.utils.molybdenum import data as molybdata

import os


def extract_data(dataset):
    graphs = []
    loader = DataLoader(dataset, shuffle=False)
    for g in loader:
        graphs.append(g)
    return graphs


class QMMineContextDataModule(pl.LightningDataModule):
    def __init__(self, data_path, split_name='sizes', batch_size=64, label_scaler=None, modification=None,
                 features='atom_ohe', *args, **kwargs):
        super(QMMineContextDataModule, self).__init__()
        self.data_path = os.path.join(os.environ['PROJECT_ROOT'], data_path)
        self.split_name = split_name
        self.batch_size = batch_size
        self.label_scaler = hydra.utils.instantiate(label_scaler)
        self.modification = modification
        self.features = features
        self.train = None
        self.val = None
        self.test = None
        self.labels_used = [5]


        self.setup()

    def setup(self, stage=None):
        self.cached_data = QM9(root=os.path.join(self.data_path, 'download'))
        #data = extract_data(full_dataset)
        #first_graph = dataset[0:1]
        # os.environ('PROJECT_ROOT') ?
        self.structure_registers = {
            'train': np.load(os.path.join(self.data_path, 'datapoint_indexes', 'sizes', 'train.npy')),
            'val': np.load(os.path.join(self.data_path, 'datapoint_indexes', 'sizes', 'val.npy')),
            'test': np.load(os.path.join(self.data_path, 'datapoint_indexes', 'sizes', 'test.npy'))
        }

        #self.train_dataset = torch.tensor(np.load())
        if self.features == 'atom_categorical':
            x = self.cached_data.data.x
            self.cached_data.data.x = torch.tensor(x[:, :5].argmax(dim=1), dtype=x.dtype, device=x.device)
        if self.features == 'atom_ohe':
            x = self.cached_data.data.x
            self.cached_data.data.x = torch.tensor(x[:, :5], dtype=x.dtype, device=x.device)
        if self.label_scaler is not None:
            self.label_scaler.fit(np.concatenate(
                            [self.cached_data[i].y for i in self.structure_registers['train'].reshape(-1)],
                            axis=0))
            cur_y = self.cached_data.data.y
            self.cached_data.data.y = torch.tensor(self.label_scaler.transform(np.array(cur_y)),
                                              dtype=cur_y.dtype, device=cur_y.device)
        # self.val_dataset = self.create_dataset(full_dataset, self.structure_registers['val'])
        # self.test_dataset = self.create_dataset(full_dataset, self.structure_registers['test'])

    def scale_used_data(self, used_data):
        #return
        if self.label_scaler is not None:
            old_ys = np.concatenate([ud.y for ud in used_data], axis=0)
            new_ys = self.label_scaler.transform(old_ys)
            new_ys_tens = torch.tensor(new_ys, dtype=used_data[0].x.dtype, device=used_data[0].x.device)
            for ud, ny in zip(used_data, new_ys_tens):
                ud.y = ny[None, self.labels_used]

    def in_context_dataloader(self, ds_name='train'):
        register = self.structure_registers[ds_name]
        used_data = [self.cached_data[i] for i in register.reshape(-1)]
        self.scale_used_data(used_data)
        return self.double_loader_trick(used_data, self.batch_size, register.shape[1])

    @staticmethod
    def double_loader_trick(data, batch_size, sequence_size):
        """
        In order to portion data into contexts, a dataloader is used. The batch_size of the first dataloader
        corresponds to the length of a sequence (context + inference), while that of the second one is the
        actual batch size. Based on the pointers of the first batch, new fields are added that help keep track
        of both the space a structure takes in a batch and in a context.
        """
        context_loader = DataLoader(data, batch_size=sequence_size, shuffle=False)
        contexts = []
        for b in context_loader:
            batch_fields = {k: v for k, v in b.items() if k not in ['batch', 'ptr']}
            batch_fields['context_num'] = b.batch
            context_datapoint = Data(**batch_fields)
            contexts.append(context_datapoint)
        full_loader = DataLoader(contexts, batch_size=batch_size, shuffle=True)
        return full_loader

    def broken_dataloader(self, ds_name='train'):
        """
        A dataloader that's broken down into a standard one: each configuration is an individual training example.
        Can be used to compare the effectiveness of training to that of an in-context model with shuffled context.
        """
        register = self.structure_registers[ds_name]
        used_data = [self.cached_data[i] for i in register.reshape(-1)]
        self.scale_used_data(used_data)
        return DataLoader(used_data, batch_size=self.batch_size, shuffle=True)

    def no_context_dataloader(self, ds_name='train'):
        """
        A dataloader that possesses the same structures for which feature inference is done, but the context
        examples are omitted. The procedure assumes that there's one dedicated example to be inferred on,
        at the end of the sequence.
        Can be used to compare the effectiveness of training to that of an in-context model without shuffling.
        """
        register = self.structure_registers[ds_name]
        used_data = [self.cached_data[i] for i in register.reshape(-1)]
        self.scale_used_data(used_data)
        return DataLoader(used_data, batch_size=self.batch_size, shuffle=True)

    def train_dataloader(self, modification=None):
        if modification is None:
            modification = self.modification
        if modification == 'no_context':
            return self.no_context_dataloader('train')
        if modification == 'breakdown':
            return self.broken_dataloader('train')
        return self.in_context_dataloader('train')

    def val_dataloader(self, modification=None):
        if modification is None:
            modification = self.modification
        if modification == 'no_context':
            return self.no_context_dataloader('val')
        if modification == 'breakdown':
            return self.broken_dataloader('val')
        return self.in_context_dataloader('val')

    def test_dataloader(self, modification=None):
        if modification is None:
            modification = self.modification
        if modification == 'no_context':
            return self.no_context_dataloader('test')
        if modification == 'breakdown':
            return self.broken_dataloader('test')
        return self.in_context_dataloader('test')


class MolybdenumDataModule(pl.LightningDataModule):
    def __init__(self, db_name, sequence_length=5, batch_size=64, label_scaler=None, modification=None,
                 datapoint_limit=None, *args, **kwargs):
        super(MolybdenumDataModule, self).__init__()
        self.db_name = db_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.label_scaler = hydra.utils.instantiate(label_scaler)
        self.modification = modification
        self.datapoint_limit = datapoint_limit
        self.train = None
        self.val = None
        self.test = None
        self.setup()

    def setup(self, stage=None):
        self.train = molybdata(self.db_name, split="train", datapoint_limit=self.datapoint_limit)
        self.val = molybdata(self.db_name, split="test", datapoint_limit=self.datapoint_limit)
        self.test = self.val

        self.label_scaler = self.label_scaler
        if self.label_scaler is not None:
            self.label_scaler.fit(np.stack(
                            [t.y for t in self.train],
                            axis=0).reshape(-1, 1))

    def scale_used_data(self, used_data):
        # return
        if self.label_scaler is not None:
            old_ys = np.stack([ud.y for ud in used_data], axis=0).reshape(-1, 1)
            new_ys = self.label_scaler.transform(old_ys)
            new_ys_tens = torch.tensor(new_ys, dtype=used_data[0].x.dtype, device=used_data[0].x.device)
            for ud, ny in zip(used_data, new_ys_tens):
                ud.y = ny
        return used_data

    def in_context_dataloader(self, data):
        return self.double_loader_trick(data, self.batch_size, self.sequence_length)

    @staticmethod
    def double_loader_trick(data, batch_size, sequence_size):
        """
        In order to portion data into contexts, a dataloader is used. The batch_size of the first dataloader
        corresponds to the length of a sequence (context + inference), while that of the second one is the
        actual batch size. Based on the pointers of the first batch, new fields are added that help keep track
        of both the space a structure takes in a batch and in a context.
        """
        context_loader = DataLoader(data, batch_size=sequence_size, shuffle=False)
        contexts = []
        for b in context_loader:
            batch_fields = {k: v for k, v in b.items() if k not in ['batch', 'ptr']}
            batch_fields['context_num'] = b.batch
            context_datapoint = Data(**batch_fields)
            contexts.append(context_datapoint)
        full_loader = DataLoader(contexts, batch_size=batch_size, shuffle=True)
        return full_loader

    def train_dataloader(self, modification=None):
        raw_data = self.train
        data = self.scale_used_data(raw_data)
        if modification is None:
            modification = self.modification
        if modification == 'breakdown':
            return DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return self.in_context_dataloader(data)

    def val_dataloader(self, modification=None):
        raw_data = self.val
        data = self.scale_used_data(raw_data)
        if modification is None:
            modification = self.modification
        if modification == 'breakdown':
            return DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return self.in_context_dataloader(data)

    def test_dataloader(self, modification=None):
        raw_data = self.test
        data = self.scale_used_data(raw_data)
        if modification is None:
            modification = self.modification
        if modification == 'breakdown':
            return DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return self.in_context_dataloader(data)


if __name__ == '__main__':
    os.chdir('../..')
    import dotenv
    from pathlib import Path
    dotenv.load_dotenv()
    data_path = os.path.join(os.environ['PROJECT_ROOT'], 'data', 'EFF')
    datamodule = MolybdenumDataModule(data_path)
    datamodule.train_dataloader()
