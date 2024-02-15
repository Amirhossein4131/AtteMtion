from data import *
from pathlib import Path
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import os
import torch_geometric
from torch.optim import Adam
from argparse import ArgumentParser

import hydra
import omegaconf
from omegaconf import OmegaConf, DictConfig
import dotenv

dotenv.load_dotenv()
PROJECT_ROOT: Path = Path(os.environ["PROJECT_ROOT"])
assert (
    PROJECT_ROOT.exists()
), "The project root doesn't exist"

OmegaConf.register_new_resolver("load", lambda x: eval(x))


@hydra.main(config_path=os.path.join('conf', 'grzegorz'), config_name='regression')
# Feel free to test the behavior of hydra: I made two equivalent versions of Your params: single_file and composed_conf
def main(cfg: DictConfig):
    # Instantiate data module
    datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Instantiate trainer with Hydra config
    trainer = pl.Trainer(**cfg.trainer)

    # Fit the model
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

