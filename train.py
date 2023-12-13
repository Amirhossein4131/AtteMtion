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



yaml_cfg = """
hydra:
    run:
        dir:
            $os.env.ROOT_PATH/hydra/experimentname
            
model:
    _target_:
        model.InContextGNN

trainer:
    max_epochs: 1000000
    log_interval: 100

"""
# This is what's passed to omegaconf. One can either use a string to create cfg, or use outside .yaml files.
# Below is how you would create cfg normally. For us, hydra does exactly that implicitly!
# cfg = OmegaConf.create(yaml_cfg)
# todo It's an example to delete after discussing

# I moved this one here, because it conflicts with @hydra.main otherwise
# I encapsulated it in function, so it's only called if __name__ indeed equals '__main__', but it's now a neat one-liner
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

