import pandas as pd
import numpy as np
import hydra
import json
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader


class DimeNetDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, test_csv, elements, label_scaler=None,
                 batch_size=1, separate_test=False, *args, **kwargs):
        super(DimeNetDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.elements = elements
        self.label_scaler = hydra.utils.instantiate(label_scaler)

        # Read atom types
        with open(elements) as file:
            self.elements = json.load(file)

        self.train_dataset = self.getdb(self, train_csv)
        if separate_test:
            self.test_dataset = self.getdb(self, test_csv)
        else:
            self.train_size = int(0.8 * len(self.train_dataset))
            self.test_size = len(self.train_dataset) - self.train_size
            self.train_dataset, self.test_dataset = random_split(self.train_dataset,
                                                             [self.train_size, self.test_size])

        if self.label_scaler is not None:
            self.label_scaler.fit(torch.stack(
                            [t.y for t in self.train_dataset],).reshape(-1, 1).numpy())

    @staticmethod
    def getdb(self, dataset_csv):
        data = pd.read_csv(dataset_csv, skipinitialspace=True)
        db = []
        for i in range(len(data)):
            cif = data['cif'][i]
            lines = cif.strip().split('\n')

            # target (y)
            prop = data['stress_Gpa'][i]

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
                              y=torch.tensor(prop, dtype=torch.float32).view(1, -1))

            db.append(data_point)
        return db

    def scale_data(self, dataset):
        yd = [d.y for d in dataset]
        yt = torch.cat(yd, dim=0)
        yt = torch.tensor(self.label_scaler.transform(yt), dtype=torch.float32)
        dataset = dataset.copy()
        for n, data in enumerate(dataset):
            dataset[n].y = yt[n:n+1]

        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    datamodule = DimeNetDataModule(train_csv="db.csv", test_csv="db.csv")
    batch = next(iter(datamodule.val_dataloader()))
    print(datamodule.val_dataloader())
