import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel as P
from importlib import import_module


data_folder = os.path.dirname(os.path.abspath(__file__))

# main에 이걸 짜야 할듯함
dataset_filenames = [
    os.path.splitext(os.path.basename(v))[0] for v in os.listdir(data_folder)
    if v.endswith('.py')]

_dataset_modules = [
    import_module(f'models.{file_name}')
    for file_name in dataset_filenames
]

class Model(nn.Module):
    def __init__(self, opt, ckp, gpu_id) -> None:
        super(Model, self).__init__()
        self.model_spec = opt["model"]
        print("Making model... {}".format(self.model_spec["type"]))

        self.device = torch.device(gpu_id)
        print(f"{gpu_id}, {self.model_spec}")
        model_type = self.model_spec.pop("type")
        for module in _dataset_modules:
            model_cls = getattr(module, model_type, None)
            print(f"{gpu_id}, {self.model_spec}")
            if model_cls is not None:
                break
        if model_cls is None:
            raise ValueError("Model not found")
        self.model = model_cls(**self.model_spec).to(self.device)

        self.load(ckp.get_path("model"), resume=opt["resume"])

    # def forward(self, sinogram, grid, scale):
    #     if self.training:
    #         return self.model(sinogram, grid, scale)
    #     else:
    #         return self.model(sinogram, grid, scale)
    def forward(self, x):
        return self.model(x)

    # is_best는 best model을 저장할 때 사용하는 변수
    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, "model_latest.pt")]
        if is_best:
            save_dirs.append(os.path.join(apath, "model_best.pt"))
        for i in save_dirs:
            torch.save(self.model.state_dict(), i)

    def load(self, apath, resume=-1, **kwargs):
        load_from = None
        if resume == 0:
            print("Load the model from {}".format(os.path.join(apath, "model_latest.pt")))
            load_from = torch.load(os.path.join(apath, "model_latest.pt"), **kwargs)
        elif resume == 1:
            print("Load the model from {}".format(os.path.join(apath, "model_best.pt")))
            load_from = torch.load(os.path.join(apath, "model_best.pt"), **kwargs)
        elif resume == 2:
            load_from = torch.load(os.path.join(apath, "model_{}.pt".format(resume)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
