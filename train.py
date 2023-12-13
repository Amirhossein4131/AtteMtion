from model import *
from data import *

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.optim import Adam
from argparse import ArgumentParser


def main(hparams):
    model = InContextGNN()
    tensorboard_logger = TensorBoardLogger("logs",
                                           name="Mo_1")

    trainer = pl.Trainer(
            accelerator=hparams.accelerator,
            devices=hparams.devices,
            max_epochs=1000000,
            logger=tensorboard_logger,
            log_every_n_steps=100
            )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
