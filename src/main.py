import os
import data
import yaml
import torch
import utils
import models
import warnings
import argparse
import torch._dynamo
from typing import Any
from trainer import Trainer
from importlib import import_module
from torch.utils.data import Dataset


torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = True

data_folder: str = os.path.join(os.path.dirname(__file__), 'data')


# load data modules
dataset_filenames: list[str] = [
    os.path.splitext(os.path.basename(v))[0] for v in os.listdir(data_folder)
    if v.endswith('.py')]

_dataset_modules: list = [
    import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


def main(rank: int, args, opt, checkpoint, train_dataset, val_dataset, ddp=True) -> None:
    if ddp:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:23456",
            world_size=opt["num_gpu"],
            rank=rank,
        )

    gpu_id: int = rank % torch.cuda.device_count()

    if checkpoint.ok:
        # loader = data.Data(config, ddp)
        _model = models.Model(opt, checkpoint, gpu_id)
        t = Trainer(opt=opt,
                    my_model=_model, 
                    ckp=checkpoint, 
                    load=args.load, 
                    gpu_id=gpu_id, 
                    ddp=ddp, 
                    test_only=False, 
                    train_dataset=train_dataset, 
                    val_dataset=val_dataset)
        while not t.terminate():
            t.train()
            t.eval()


def load_dataset(opt) -> tuple[Dataset, Dataset]:
    custom_dataset: Any = opt["datasets"]["train"]["type"]
    for module in _dataset_modules:
        dataset_cls: Dataset = getattr(module, custom_dataset, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError("Dataset not found")
    train_dataset: Dataset = dataset_cls(opt, mode='train')
    print(f"train {custom_dataset} loaded")
    val_dataset: Dataset = dataset_cls(opt, mode='val')
    print(f"val dataset {custom_dataset} loaded")
    return train_dataset, val_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="/workspace/options/MIMOUNet_opts")
    parser.add_argument("--save", type=str, default="MIMOUNet_test")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--n_gpus", type=int, default=4)
    args: argparse.Namespace = parser.parse_args()

    with open(os.path.join(args.opt + ".yaml"), "r") as f:
        opt: dict = yaml.load(f, Loader=yaml.FullLoader)
        print("option file loaded, opt_path: {}".format(os.path.join("options", args.opt + ".yaml")))
        opt["num_gpu"] = args.n_gpus


    train_dataset, val_dataset = load_dataset(opt)

    checkpoint = utils.checkpoint(opt, args.load, args.save)
    if opt["num_gpu"] > 1:
        torch.multiprocessing.spawn(main, args=(args, opt, checkpoint, train_dataset, val_dataset), nprocs=opt["num_gpu"])
    else:
        main(0, args, opt, checkpoint, train_dataset, val_dataset, ddp=False)
