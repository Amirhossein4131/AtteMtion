from model import * # please avoid the import * clause
from data import *

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()
    return args

@hydra.main(config_path=os.path.join('conf', 'amir'), config_name='composed_conf.yaml')
# Feel free to test the behavior of hydra: I made two equivalent versions of Your params: single_file and composed_conf
def main(cfg: DictConfig):
    args = parse_args()
    model = hydra.utils.instantiate(cfg.model)
    tensorboard_logger = TensorBoardLogger("logs",
                                           name="Mo_1")

    trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=cfg.trainer.max_epochs,
            logger=tensorboard_logger,
            log_every_n_steps=cfg.trainer.log_interval
            )
    trainer.fit(model)


if __name__ == "__main__":
    main()

