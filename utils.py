import os
import gc
import time
import math
import torch
import shutil
import datetime
import numpy as np
import lr_scheduler
import pandas as pd
import torch.fft as fft
import torch.optim as optim
from torch import nn, Tensor
from models.EDVR import EDVR
import torch.nn.functional as F
import torch.distributed as dist
from models.MIMOUNet import MIMOUNet
from torch.utils.data import Sampler
from models.ReBotNet import ReBotNet
from typing import Any, Iterator, Union
from monai.losses.ssim_loss import SSIMLoss
from models.KernelWizard import KernelWizard
from torch.autograd import Variable, detect_anomaly
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas: int = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank: int = dist.get_rank()
        self.dataset: Any = dataset
        self.num_replicas: int | Any = num_replicas
        self.rank: int | Any = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size: int = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices: list[int] = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples: int = len(indices)             # true value without extra samples

        self.shuffle: bool = shuffle
        self.seed: int = seed

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices: list = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch: int = epoch


class Timer:
    def __init__(self) -> None:
        self.v: float = time.time()

    def s(self) -> None:
        self.v: float = time.time()

    def t(self) -> float:
        return time.time() - self.v


class Averager:
    def __init__(self) -> None:
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0) -> None:
        self.v: Any = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self) -> float|Any:
        return self.v


def compute_num_params(model, text=False) -> str | int:
    num_params = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if num_params >= 1e6:
            return "{:.1f}M".format(num_params / 1e6)
        else:
            return "{:.1f}K".format(num_params / 1e3)
    else:
        return num_params


# test_only는 파일 저장 안하고 결과만 볼거면 사용하는 변수인듯 함
# val할 때는 test_only=True로 설정, test할 때는 False로 설정
class checkpoint:
    def __init__(self, config, load="", save="", test_only=False) -> None:
        """
        Stores checkpoints for training and testing
        config: configuration dictionary
        load: load directory, used for resuming training or loading models for testing along
        save: save directory, used for saving models
        test_only: distinguish between resuming training and testing
        
        """
        self.ok = True
        self.train_log = Tensor()
        self.val_log = Tensor()
        self.test_only: bool = test_only

        now: str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        # load가 주어지지 않은 경우
        if load == "":
            # save가 명시되지 않으면
            if not save:
                # 현재 시간을 받아와서 폴더 생성
                save: str = now
            # self.dir은 experiment 폴더에 save 폴더를 만들어서 저장
            self.dir: str = os.path.join("./experiment", save)
        # load가 무언가 주어지면
        # 이 경우 이미 진행되던 학습을 이어서 할 때 사용하는 듯 함
        else:
            # load할 폴더를 찾아서 dir에 저장
            self.dir = os.path.join("./experiment", load)

            # load할 폴더 내의 파일들을 가져옴
            # 이건 training시에 resume을 할 때 가져올 때 사용하는 듯
            # test only 일 때는 이 결과를 load하는 것보다 새로 만드는 게 맞을 것 같음
            if os.path.exists(self.dir) and not test_only:
                self.train_log = torch.load(self.get_path("loss_log.pt"))
                self.val_log = torch.load(self.get_path("val_log.pt"))

        # log 및 결과 보관하는 폴더 위치
        print("experiment directory is {}".format(self.dir))

        if not self.test_only:
            # self.dir, self.dir/model, self.dir/results 폴더 생성
            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(self.get_path("model"), exist_ok=True)
            os.makedirs(self.get_path("results"), exist_ok=True)

            # options 저장
            with open(os.path.join(self.dir, "options.txt"), "w") as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
            print("options saved successfully.")
        

    def get_path(self, *subdir) -> str:
        return os.path.join(self.dir, *subdir)

    def add_train_log(self, log) -> None:
        self.train_log: Tensor = torch.cat([self.train_log, log])

    def add_val_log(self, log) -> None:
        self.val_log: Tensor = torch.cat([self.val_log, log])

    def save(self, trainer, epoch, is_best=False, n_gpus=1) -> None:
        # model은 model 폴더의 __init__.py에 저장된 save를 사용
        # save(self, apath, epoch, is_best=False)
        # apath로 입력된 경로에 model_latest.pt를, is_best가 True이면 model_best.pt도 함께 저장
        if n_gpus > 1:
            trainer.model.module.save(self.get_path("model"), epoch, is_best=is_best)
        else:
            trainer.model.save(self.get_path("model"), epoch, is_best=is_best)
        # optimizer는 utils.py에 저장된 save를 사용
        trainer.optimizer.save(self.dir)
        # loss_log, rmse_log 저장
        torch.save(self.train_log, os.path.join(self.dir, "loss_log.pt"))
        torch.save(self.val_log, os.path.join(self.dir, "val_log.pt"))


# def calc_psnr_per_slice(test, ref, data_dim):
#     # mean only the 2D dims
#     mse = ((test - ref) ** 2).mean([-2, -1])
#     return 20 * torch.log10(ref.max() / torch.sqrt(mse)).cpu().item()


def calc_psnr_per_slice(test:Tensor, ref:Tensor, max_value:float=1.0) -> float:
    """Calculate PSNR per a frame and average them."""
    mse: float = ((test - ref) ** 2).mean([-4, -2, -1]) # Assume (B, C, D, H, W) format video.
    # psnr: float = 20 * torch.log10(ref.max(-1).values.max(-1).values / torch.sqrt(mse)).mean().cpu().item()
    psnr: float = 20 * torch.log10(max_value / torch.sqrt(mse)).mean().cpu().item()
    
    del test, ref
    torch.cuda.empty_cache()
    gc.collect()
    return psnr


def calc_rmse_per_slice(test:Tensor, ref:Tensor) -> float:
    """Calculate RMSE per a frame and average them."""
    # Assume (B, C, D, H, W) format video.
    mse: float = ((test - ref) ** 2).mean([-4, -2, -1])

    del test, ref
    torch.cuda.empty_cache()
    gc.collect()

    # compute slicewise rmse
    return torch.sqrt(mse).mean().cpu().item()


def calc_ssim(img1: Tensor, img2: Tensor, data_range, window_size: int=11, channel: int=3, size_average: bool=True) -> float:
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.shape) == 2:
        h, w = img1.shape
        if type(img1) == Tensor:
            img1: Tensor = img1.view(1, 1, h, w)
            img2: Tensor = img2.view(1, 1, h, w)
        else:
            img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :])
            img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :])
    elif len(img1.shape) == 3:
        if type(img1) == Tensor:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        else:
            img1 = torch.from_numpy(img1[np.newaxis, :, :, :])
            img2 = torch.from_numpy(img2[np.newaxis, :, :, :])

    window: Tensor = create_window(window_size, channel)
    window: Tensor = window.type_as(img1)

    mu1: Tensor = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2: Tensor = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2: Tensor = mu1*mu2

    #  sigma**2 = E[mu**2] = E[mu]**2
    sigma1_sq: Tensor = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq: Tensor = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12: Tensor = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1, C2 = (0.0*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.012, 0.032

    ssim_map: Any = ((2*mu1_mu2+C1)(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)(sigma1_sq+sigma2_sq+C2))
    if size_average:
        # returns single value
        return ssim_map.mean().item()
    else:
        # returns Nx1 tensor containing ssim of N image pairs
        return ssim_map.mean(1).mean(1).mean(1).item()


# create 1D gaussian kernel centered at window_size // 2
# normalized by the sum of the kernel for probability sum to 1
def gaussian(window_size, sigma) -> Tensor:
    gauss = Tensor([np.exp(-(x - window_size // 2)**2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# create 2D gaussian kernel by outer product of 1D gaussian kernel
# out_channel = channel, in_channel = 1, kernel_size = window_size
# in channel is fixed to 1 because it is in_channel/groups for conv2d
# ssim computes the conv2d of img1 and img2 for each channels using the group parameter (groups = channel)
def create_window(window_size, channel) -> Tensor:
    _1D_window: Tensor = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window: Tensor = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def normalize(img1: Tensor, mean: float, std: float) -> Tensor:
    img1 = (img1 - mean) / std
    return img1


def denormalize(img: Tensor, mean: float, std: float) -> Tensor:
    img1 = img * std + mean
    return img1

# custom scheduler
def make_scheduler(scheduler_spec, last_epoch, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    # pop을 해야 **kwargs에 다른 인자들이 제대로 들어가게 됨
    scheduler_type = scheduler_spec.pop('type')
    scheduler_spec['last_epoch'] = last_epoch
    if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
        scheduler = lr_scheduler.MultiStepRestartLR(
                    optimizer, **scheduler_spec)
    elif scheduler_type == 'CosineAnnealingRestartLR':
        scheduler = lr_scheduler.CosineAnnealingRestartLR(
                    optimizer, **scheduler_spec)
    elif scheduler_type == 'CosineAnnealingWarmUpRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmUpRestarts(
                    optimizer, **scheduler_spec)
    elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            scheduler = lr_scheduler.CosineAnnealingRestartCyclicLR(
                    optimizer, **scheduler_spec)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, **scheduler_spec)
    elif scheduler_type == 'CosineAnnealingLRWithRestart':
        scheduler = lr_scheduler.CosineAnnealingLRWithRestart(
                    optimizer, **scheduler_spec)
    elif scheduler_type == 'LinearLR':
        scheduler = lr_scheduler.LinearLR(
                    optimizer, scheduler_spec['total_iter'])
    elif scheduler_type == 'VibrateLR':
        scheduler = lr_scheduler.VibrateLR(
                    optimizer, scheduler_spec['total_iter'])
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, **scheduler_spec)
    elif scheduler_type == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, **scheduler_spec)
    else:
        raise NotImplementedError(
            f'Scheduler {scheduler_type} is not implemented yet.')

    return scheduler

# optimizer opt 받아서 optimizer 만들기
def make_optimizer(optim_spec, target) -> torch.optim.Optimizer:
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer: dict[str|float] = {"lr": optim_spec["lr"], "weight_decay": optim_spec["weight_decay"]}

    if optim_spec["type"] == "SGD":
        optimizer_class = optim.SGD
    elif optim_spec["type"] == "ADAM":
        optimizer_class = optim.Adam
    elif optim_spec['type'] == 'AdamW':
        optimizer_class = optim.AdamW
    elif optim_spec["type"] == "RMSprop":
        optimizer_class = optim.RMSprop
    elif optim_spec["type"] == "RADAM":
        optimizer_class = optim.RAdam

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs) -> None:
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def save(self, save_dir) -> None:
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir) -> None:
            self.load_state_dict(torch.load(self.get_dir(load_dir)))

        def get_dir(self, dir_path) -> str:
            return os.path.join(dir_path, "optimizer.pt")

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    return optimizer


# applying lr warmup
def lr_warmup(training_details, epoch, optimizer) -> Any:
    warmup_iter: Any = training_details['warmup_iter']
    if epoch < warmup_iter:
        init_lr: Any = optimizer.param_groups['initial_lr']
        lr: list = [v/warmup_iter*epoch for v in init_lr]
        for pg, warmed_lr in zip(optimizer.param_groups, lr):
            pg['lr'] = warmed_lr
    return optimizer


def get_model_name(model: Union[DDP, nn.Module]) -> Union[str, None]:
    """
    Get the model name from the model object.
    
    Args:
        model (Union[DistributedDataParallel, nn.Module]): The model object.
    
    Returns:
        name (Union[str, None]): The model name as a string, or None if not matched.
    """
    # Extract the inner model object safely
    if isinstance(model, DDP):
        inner: nn.Module = getattr(model.module, "model", model.module)
    else:
        inner: nn.Module = getattr(model, "model", model)

    # Use match-case if available (Python 3.10+), else fallback
    match inner:
        case ReBotNet():
            return "ReBotNet"
        case EDVR():
            return "EDVR"
        case KernelWizard():
            return "KernelWizard"
        case MIMOUNet():
            return "MIMOUNet"
        case _:
            return None


# for clipping videos during test time
def test_video(lq: Tensor, model: Union[DDP, nn.Module], gpu_id: int, args: dict, gt: Union[Tensor, None]=None) -> Tensor|Any:
    '''
    Args:
        lq (Tensor): low-quality video tensor of shape (B, C, D, H, W)
        model (Union[DDP, nn.Module]): The model object.
        gpu_id (int): GPU ID for the model.
        args (dict): Arguments for the model.
        gt (Union[Tensor, None]): Ground truth tensor of shape (B, C, D, H, W) or None.
    
    Returns:
        output (Tensor|Any): The output tensor after processing.
    '''

    num_frame_testing: Any = args["tile"][0]

    model_name: str|None = get_model_name(model)

    # num_frame_testing이 존재하면 frame을 clip함
    if num_frame_testing:
        # test as multiple clips if out-of-memory

        # DVD의 경우 args.scale=1
        # videoSR에 사용하는 parameter
        # sf = args["scale"]

        # DVD의 경우 args.tile_overlap을 [2, 20, 20]로 둠
        num_frame_overlapping: Any = args["tile_overlap"][0]
        not_overlap_border = False
        print(lq.size())
        while lq.dim() > 5:
            lq = lq.squeeze(0)
            print(lq.size())

        if lq.dim() == 5:
            b, c, d, h, w = lq.size()
        elif lq.dim() == 4:
            b, c, h, w = lq.size()
            d = 1

        # DVD의 경우 args.nonblind_denoising=False
        # c = c - 1 if args["nonblind_denoising"] else c

        # Determine stride based on model type
        if model_name in ["ReBotNet", "EDVR"]:
            stride: int = 1  # These models operate per-frame
        else:
            stride: int = num_frame_testing - num_frame_overlapping  # Sliding window for other models

        # Compute test frame index list depending on model
        match model_name:
            case "ReBotNet":
                # ReBotNet can't predict the first frame unless a previous GT frame is given
                d_idx_list: list[int] = list(range(0, d - 1, stride))
            case "EDVR":
                # EDVR requires 5 input frames to predict the middle one
                d_idx_list: list[int] = list(range(0, d - 5, stride)) + [d - 5]
            case _:
                # Models with identical I/O dimensions
                d_idx_list: list[int] = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]

        print(d_idx_list)

        # Preallocate output tensors (E = enhanced frame; W = frame-wise weight map)
        match model_name:
            case "ReBotNet":
                # Predicts frames starting from index 1
                E: Tensor = torch.zeros_like(lq[:, :, 1:, ...]).cpu()
                W: Tensor = torch.zeros(b, 1, d - 1, 1, 1).cpu()
            case "EDVR":
                # EDVR predicts from 5-frame inputs, resulting in d-4 outputs
                E: Tensor = torch.zeros_like(lq[:, :, 2:-2, ...]).cpu()
                W: Tensor = torch.zeros(b, 1, d - 4, 1, 1).cpu()
            case _:
                # Generic output and weight tensors (scaled outputs)
                E: Tensor = torch.zeros_like(lq).cpu()
                W: Tensor = torch.zeros(b, 1, d, 1, 1).cpu()



        # d_idx_list contains the starting index of each video clip
        for d_idx in d_idx_list:
            start_time = time.time()

            # Slice low-quality input clip
            lq_clip: Tensor = lq[:, :, d_idx:d_idx + num_frame_testing, ...]

            # For KernelWizard, also slice the corresponding ground truth
            if model_name == "KernelWizard":
                gt_clip: Tensor = gt[:, :, d_idx:d_idx + num_frame_testing, ...]

            # ReBotNet uses the previous prediction as the first frame in the next clip
            if model_name == "ReBotNet" and d_idx > 0:
                lq_clip[:, :, 0, ...] = out_clip  # Use last prediction as first frame

            # Perform inference
            out_clip: Tensor = (
                test_clip(lq_clip, model, gpu_id, args, gt_clip)
                if model_name == "KernelWizard"
                else test_clip(lq_clip, model, gpu_id, args)
            )

            # Create a binary mask (with 1s) to average overlapping predictions
            valid_len = min(num_frame_testing, d)
            out_clip_mask: Tensor = torch.ones((b, 1, valid_len, 1, 1), device=gpu_id)

            # Optionally remove border overlaps from output and mask
            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, :, -num_frame_overlapping // 2:, ...] *= 0
                    out_clip_mask[:, :, -num_frame_overlapping // 2:, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, :, :num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, :, :num_frame_overlapping // 2, ...] *= 0

            # Accumulate the enhanced predictions and their weights
            if model_name in ["ReBotNet", "EDVR"]:
                E[:, :, d_idx, ...].add_(out_clip.cpu())
                W[:, :, d_idx, ...].add_(out_clip_mask[:, :, 1, ...].cpu())
            else:
                E[:, :, d_idx:d_idx + num_frame_testing, ...].add_(out_clip.cpu())
                W[:, :, d_idx:d_idx + num_frame_testing, ...].add_(out_clip_mask.cpu())

            print(f"[INFO] Processed frame index {d_idx}, time taken: {time.time() - start_time:.2f}s")

        # Normalize output by dividing with the weight mask
        output: Tensor = E.div_(W)

    else:
        # test as one clip (the whole video) if you have enough memory
        window_size: Any = args["window_size"]
        d_old: Any = lq.size(2)
        d_pad:Any = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq: Tensor = torch.cat([lq, torch.flip(lq[:, :, -d_pad:, ...], [2])], 2) if d_pad else lq
        output = test_clip(lq, model, args)
        output = output[:, :, :d_old, ...]

    del out_clip, out_clip_mask, lq, lq_clip, E, W
    torch.cuda.empty_cache()
    gc.collect()
    return output


def test_clip(lq: Tensor, model: Union[DDP, nn.Module], gpu_id: Union[str, int], args: dict, gt: Union[Tensor, None]=None) -> Tensor|Any:
    '''
    Args:
        lq (Tensor): low-quality video tensor of shape (B, C, D, H, W)
        model (Union[DDP, nn.Module]): The model object.
        gpu_id (Union[str, int]): GPU ID for the model.
        args (dict): Arguments for the model.
        gt (Union[Tensor, None]): Ground truth tensor of shape (B, C, D, H, W) or None.
    
    Returns:
        output (Tensor|Any): The output tensor after processing.
    '''

    # DVD의 경우 args.scale=1
    # videoSR에 사용하는 parameter
    # sf = args["scale"]

    # DVD의 경우 args.window_size=[6, 8, 8]
    if args.get("window_size"):
        window_size: Any = args["window_size"]

    model_name: str|None = get_model_name(model)

    # DVD의 경우 args.tile=[12, 256, 256]
    # 256으로 clip을 하여 계산하겠다는 의미
    size_patch_testing: Any = args["tile"][1]

    # clip size가 window_size로 나누어져야함
    # assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        # DVD의 경우 args.tile_overlap=[2, 20, 20]
        overlap_size: Any = args["tile_overlap"][1]


        not_overlap_border = False

        # test patch by patch
        # Unpack dimensions of lq tensor depending on its rank
        match lq.dim():
            case 5:
                # 5D tensor: (B, C, D, H, W) → Video or multi-frame volume
                b, c, d, h, w = lq.size()
            case 4:
                # 4D tensor: (B, C, H, W) → Single frame input, treat D=1
                b, c, h, w = lq.size()
                d = 1
            case _:
                raise ValueError(f"Unsupported input dimension: {lq.dim()}")


        # c = c - 1 if args["nonblind_denoising"] else c

        # stride = 256 - 20 = 236
        stride: int = size_patch_testing - overlap_size

        # h, w에 대한 patch의 시작 지점을 정의
        h_idx_list: list[int] = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list: list[int] = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]

        # lq와 같은 크기의 E, W 선언
        # E는 잘린 clip들을 합쳐서 복원할 원본 img
        # W는 overlap된 영역의 pixel값들을 나눠주기 위한 mask, overlap이 없는 경우 1
        # E = torch.zeros(b, c, 1, h*sf, w*sf) for the ReBotNet and EDVR
        if model_name in ["ReBotNet", "EDVR"]:
            E: Tensor = torch.zeros_like(lq[:, :, 1, ...]).cpu()
        else:
            E: Tensor = torch.zeros_like(lq).cpu()

        W: Tensor = torch.zeros_like(E).cpu()
        output = None

        # 각 h, w에 대한 시작지점에 대해
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                # lq를 h_idx, w_idx부터 size_patch_testing만큼 clip함
                in_patch: Tensor = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                gt_patch: Tensor = gt[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing] if gt is not None else None
                
                # 이를 model에 넣어서 out_patch를 얻음
                # out_patch는 (b, c, h, w)의 형태
                # out_patch = model(in_patch).detach().cpu()
                match model_name:
                    case "KernelWizard":
                        out_patch: Tensor = model(gt_patch, in_patch).detach()
                    case "MIMOUNet":
                        out_patch: Tensor = model(in_patch).detach()[-1] # The last path has the original resolution.
                    case _:
                        out_patch: Tensor = model(in_patch).detach()
                
                # out_patch_mask는 out_patch와 같은 크기의 1 tensor
                out_patch_mask: Tensor = torch.ones_like(out_patch)

                # 기본적으로 not_overlap_border는 size_patch_testing이 0이 아니면 True(H, W 방향 clip을 할거면 True)
                # not_overlap_border인 경우, 전체 img에서 가장자리에 있는 patch들을 제외하고는 각 patch들의 가장자리를 overlap_size//2만큼 0으로 만들어서
                # 나중에 img 만큼의 영역을 더할 때, overlap되는 영역이 발생하지 않도록 함
                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                # E, W에 out_patch, out_patch_mask를 더함
                # E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                E[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch.cpu())
                # W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
                W[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch_mask.cpu())
        
        # E를 W로 나눠서 overlap을 고려한 output을 얻음
        output: Tensor = E.div_(W)

    # size_patch_testing이 0이면 clip을 하지 않고 한 번에 계산
    else:
        _, _, _, h_old, w_old = lq.size()

        # padding을 해서 window_size로 나누어 떨어지도록 함
        # h_pad, w_pad는 window_size로 나누어 떨어지지 않는 나머지
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        # h_pad, w_pad만큼 끝쪽 image를 복사해서 padding
        lq: Tensor = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq: Tensor = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        # output = model(lq).detach().cpu()
        output = model(lq).detach()

        # output = output[:, :, :, :h_old*sf, :w_old*sf]
        output = output[..., :h_old, :w_old]
    
    del out_patch, out_patch_mask, lq, E, W
    torch.cuda.empty_cache()
    gc.collect()
    return output
