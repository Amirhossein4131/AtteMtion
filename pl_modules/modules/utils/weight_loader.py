import torch
import pytorch_lightning as pl

from typing import Union, List
import os
from os import PathLike
from omegaconf import ListConfig


class ModelLoader:
    def __init__(self, file_path: PathLike, src: Union[str, List[str]], tgt: Union[str, List[str]], freeze: bool=True):
        ''' A class used to load existing state dicts to a submodule of the model during initialization.

        Args:
        file_path - checkpoint path (relative to the PROJECT_ROOT)
        src - the path within the hierarchy of the checkpoint that points to the state dict of chosen submodule
        tgt - the path within the hierarchy of pl.LightningModule that points to the submodule of identical architecture
        freeze - whether to freeze the submodule's parameters after loading
        '''

        self.file_path = os.path.join(os.environ['PROJECT_ROOT'], file_path)
        self.src = '.'.join(src) if isinstance(src, List) or isinstance(src, ListConfig) else src
        self.tgt = '.'.join(tgt) if isinstance(tgt, List) or isinstance(tgt, ListConfig) else tgt

    def get_state_dict(self):
        sd = torch.load(self.file_path)
        if 'state_dict' in sd.keys():
            sd = sd['state_dict']
        return sd

    def translate_state_dict(self, sd):
        return {k.replace(self.src, self.tgt, 1): v for k, v in sd.items() if k.startswith(self.src)}

    def apply(self, model, path=None):
        if path is None:
            path = self.file_path
        sd = self.get_state_dict()
        sd = self.translate_state_dict(sd)
        model.load_state_dict(sd, strict=False)
        print(f"\n\nThe weights were loaded from {self.src} to {self.tgt}\n\n")


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()

    class ParentModel(pl.LightningModule):
        def __init__(self):
            super(ParentModel, self).__init__()
            self.layer = torch.nn.Linear(1, 5)



    class ChildModel(pl.LightningModule):
        def __init__(self, loader: Union[ModelLoader, None]=None):
            super(ChildModel, self).__init__()
            layer = torch.nn.Linear(1, 5)
            self.child_layer = torch.nn.Module()
            self.child_layer.layer = layer

        def load(self):
            self.loader.apply()


    pm = ParentModel()
    torch.save(pm.state_dict(), os.path.join(os.environ['PROJECT_ROOT'], 'code_tests', 'model_loader', 'ckpt'))


    cm = ChildModel()
    print('Parent:', pm.layer.bias)
    print('Child before:', cm.child_layer.layer.bias)

    loader = ModelLoader((os.path.join('code_tests', 'model_loader', 'ckpt')),
        src=['layer'],
        tgt=['child_layer', 'layer'])

    # BTW, if you pass the ModelLoader to your lightning module, the translation to hydra of this loader definition would look like:
    '''
    loader:
      __target__: pl_modules.modules.utils.weight_loader.ModelLoader
      file_path: ${oc.env:PROJECT_ROOT}/code_tests/model_loader/ckpt
      src:
        - layer
      tgt:
        - child_layer
        - layer 
    
    
    '''

    loader.apply(cm)
    #cm.child_layer.load_state_dict(pm.state_dict())
    print('Child after:', cm.child_layer.layer.bias)
    