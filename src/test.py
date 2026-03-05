import os
import data
import yaml
import torch
import utils
import models
import argparse
from tqdm import tqdm
import torch.nn as nn
from trainer import Trainer
from importlib import import_module


data_folder: str = os.path.join(os.path.dirname(__file__), 'data')


# load data modules
dataset_filenames: list[str] = [
    os.path.splitext(os.path.basename(v))[0] for v in os.listdir(data_folder)
    if v.endswith('.py')]

_dataset_modules: list = [
    import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]

torch._dynamo.config.suppress_errors = True


def main(rank: int, args, opt, checkpoint, test_dataset, ddp=True) -> None:
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
        t = Trainer(opt, _model, checkpoint, args.load, gpu_id, ddp, True, None, None, test_dataset)
        t.eval()

def load_dataset(opt):
    custom_dataset = opt["datasets"]["train"]["type"]
    for module in _dataset_modules:
        dataset_cls = getattr(module, custom_dataset, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError("Dataset not found")
    test_dataset = dataset_cls(opt, mode='test')
    print(f"test dataset {custom_dataset} loaded")
    return test_dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="/workspace/options/MIMOUNet_opts")
    parser.add_argument("--load", type=str, default="MIMOUNet_test")
    parser.add_argument("--n_gpus", type=int, default=1)
    args: argparse.Namespace = parser.parse_args()

    with open(os.path.join(args.opt + ".yaml"), "r") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
        print("option file loaded, opt_path: {}".format(os.path.join("options", args.opt + ".yaml")))
        opt["num_gpu"] = args.n_gpus


    test_dataset = load_dataset(opt)

    checkpoint = utils.checkpoint(opt, args.load, "", test_only=False) # calc test metrics only or not
    if opt["num_gpu"] > 1:
        torch.multiprocessing.spawn(main, args=(args, opt, checkpoint, test_dataset), nprocs=opt["num_gpu"])
    else:
        main(0, args, opt, checkpoint, test_dataset, ddp=False)
