import os
import gc
import io
import math
import time
import utils
import torch
import losses
import random
import numpy as np
from tqdm import tqdm
from torch import nn, Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Callable
from torch.optim import Optimizer
from importlib import import_module
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, BatchSampler
from models import ReBotNet, EDVR, KernelWizard, MIMOUNet
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, opt:dict, my_model:nn.Module, ckp:bool, load:str, gpu_id:int, ddp:bool, test_only:bool=False, train_dataset:Dataset=None, val_dataset:Dataset=None, test_dataset:Dataset=None) -> None:
        self.opt: dict = opt
        self.data_spec: Any = opt["datasets"]
        self.training_spec: Any = opt["train"]
        self.testing_spec: Any = opt["test"]
        self.scaler = GradScaler()
        self.ddp: bool = ddp
        self.ckp: bool = ckp
        self.gpu_id: int = gpu_id
        self.test_only: bool = test_only

        # Determine model type
        match my_model.model:
            case ReBotNet.ReBotNet():
                self.model_type = "ReBotNet"
            case EDVR.EDVR():
                self.model_type = "EDVR"
            case KernelWizard.KernelWizard():
                self.model_type = "KernelWizard"
            case MIMOUNet.MIMOUNet():
                self.model_type = "MIMOUNet"
            case _:
                self.model_type = "Default"

        # Assign loss function if training
        if not self.test_only:
            pixel_type: str = self.opt['train']['pixel_opt']['type']
            pixel_cfg: dict = self.opt['train']['pixel_opt']
            
            match pixel_type:
                case "L1Loss":
                    self.loss = nn.L1Loss()
                case "CharbonnierLoss":
                    self.loss = losses.CharbonnierLoss()
                case "CharbonnierSSIMLoss":
                    self.loss = losses.CharbonnierSSIMLoss(
                        spatial_dims=self.opt['data_dim'],
                        lamda=0.1
                    )
                case "MSFRLoss":
                    cont_loss_class = getattr(nn, pixel_cfg['content_loss'])
                    self.loss = losses.MSFRLoss(
                        content_loss=cont_loss_class(),
                        lamda=pixel_cfg['lamda'],
                        scales=pixel_cfg['scales']
                    )
                case _:
                    raise ValueError(f"Unknown pixel loss type: {pixel_type}")

            if self.ddp:
                # num_replica = world_size, rank=rank is automatifcally set by torch.distributed.launch
                self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
                self.test_sampler = DistributedSampler(val_dataset, shuffle=False)
                # self.model = DDP(torch.compile(my_model, dynamic=True), device_ids=[self.gpu_id], find_unused_parameters=True)
                self.model = DDP(my_model, device_ids=[self.gpu_id], find_unused_parameters=True)
                torch.set_float32_matmul_precision("high")
            else:
                self.train_sampler = None
                self.test_sampler = None
                # self.model: Callable = torch.compile(my_model, dynamic=True)
                self.model = my_model
                torch.set_float32_matmul_precision("high")
            
            # opt에서 optimizer 정보 받아와서 optimizer 만들기
            self.optimizer: Optimizer = utils.make_optimizer(self.training_spec["optimizer"], self.model)

            # 학습하던 정보가 있다면 가져오기
            # load는 test할 때만 사용하는 것으로, train때는 save만 사용(save는 Trainer에서 인자로 받지는 않음)
            if load != "":
                self.optimizer.load(ckp.dir)
                if self.ddp:
                    self.model.module.load(ckp.get_path("model"), resume=0)
                else:
                    self.my_model.load(ckp.get_path("model"), resume=0)

            # self.scheduler = MultiStepLR(
            #     self.optimizer,
            #     milestones=opt["train"]["milestones"],
            #     gamma=config["optimizer"]["gamma"],
            #     last_epoch=len(ckp.train_log) - 1,
            # )

            self.scheduler = utils.make_scheduler(self.training_spec['scheduler'], len(ckp.train_log)-1, self.optimizer)
            self.loss_list = []

            print("total number of parameter is {:,}".format(sum(p.numel() for p in self.model.parameters())))
            self.test_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.data_spec["val"]["batch_size"],
                sampler=self.test_sampler,
                num_workers=self.data_spec["val"]["num_workers"],
                shuffle=False,
                pin_memory=True
            )

            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.data_spec["train"]["batch_size"],
                sampler=self.train_sampler,
                num_workers=self.data_spec["train"]["num_workers"],
                shuffle=(self.train_sampler is None),
                pin_memory=True,
            )

        else: # test_only
            if self.ddp:
                # num_replica = world_size, rank=rank is automatifcally set by torch.distributed.launch
                self.test_sampler = DistributedSampler(test_dataset, shuffle=False)
                # self.model = DDP(torch.compile(my_model, dynamic=True), device_ids=[self.gpu_id])
                self.model = DDP(my_model, device_ids=[self.gpu_id], find_unused_parameters=True)
                torch.set_float32_matmul_precision("high")
            else:
            # self.model = my_model
                self.test_sampler = None
                # self.model = torch.compile(my_model, dynamic=True)
                self.model = my_model
                torch.set_float32_matmul_precision("high")
            
            # load best model
            if load != '':
                if self.ddp:
                    self.model.module.load(ckp.get_path("model"), resume=1)
                else:
                    self.model.load(ckp.get_path("model"), resume=1)

            self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_spec["test"]["batch_size"],
            sampler = self.test_sampler,
            num_workers=self.data_spec["test"]["num_workers"],
            shuffle=False,
            pin_memory=True,
            )


    def train(self) -> None:
        logf: io.TextIOWrapper = open('/workspace/logs/trainLog.txt', 'a+')
        # scheduler의 last_epoch를 가져와서 epoch로 사용
        # last_epoch은 scheduler.step()이 불릴 때마다 update됨
        epoch = self.scheduler.last_epoch
        # train_log에 loss 저장
        self.ckp.add_train_log(torch.zeros(1))
        # learning rate warmup
        self.optimizer = utils.lr_warmup(self.training_spec['details'], epoch, self.optimizer)
        # 지난 iteration의 learning rate 가져오기
        learning_rate = self.scheduler.get_last_lr()[0]
        self.model.train()
        train_loss = utils.Averager()
        timer = utils.Timer()


        if self.ddp:
            self.train_sampler.set_epoch(epoch)
        for iters, (img_lq, img_gt) in enumerate(self.train_loader):
            # for batch, (sino, img, patch_coord) in enumerate(tqdm(self.loader_train, disable=(self.gpu_id != 0))):
            img_lq, img_gt = img_lq.cuda(self.gpu_id), img_gt.cuda(self.gpu_id)

            self.optimizer.zero_grad()

            if self.training_spec['details']["AMP"]:
                with autocast(device_type="cuda"):
                    if self.model_type == "KernelWizard":
                        recon_img: Tensor = self.model(img_gt, img_lq)
                    else:
                        recon_img: Tensor = self.model(img_lq) # [batch, ch, h, w] or [batch, ch, d, h, w]

                    # Some models have extraordinary architectures that require a different input shape or type.
                    match self.model_type:
                        case "ReBotNet":
                            loss: Tensor = self.loss(recon_img, img_gt[:, :, 1, :, :])
                        case "EDVR":
                            # EDVR predicts a middle frame.
                            loss: Tensor = self.loss(recon_img, img_gt[:, :, 2, ...])
                        case "KernelWizard":
                            loss: Tensor = self.loss(recon_img, img_lq)  # recon_img is not sharp. It is blur.
                        case _:
                            loss: Tensor = self.loss(recon_img, img_gt)


                print(f'Is this loss tensor requires autograd?: {loss.requires_grad}', file=logf, flush=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.training_spec['details']["use_grad_clip"]:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.training_spec['details']['grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon_img = self.model(img_lq)
                loss = self.loss(recon_img, img_gt)
                loss.backward()
                if self.training_spec['details']["use_grad_clip"]:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
                self.optimizer.step()

            if self.gpu_id == 0:
                self.loss_list.append(loss.item())
                torch.save(self.loss_list, f='/workspace/logs/TrainLosses.pth')
            print(f'Rank: {self.gpu_id}, Epochs: {epoch + 1}, iterations: {iters}, Loss: {loss.item()}', file=logf, flush=True)

            train_loss.add(loss.item(), img_lq.shape[0])
            dist.barrier()
            print(file=logf, flush=True)

        loss_value: torch.Tensor = torch.tensor(train_loss.item(), device=self.gpu_id)

        if self.ddp:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        if self.gpu_id == 0:
            train_loss_avg = loss_value.item() / self.opt["num_gpu"]
            print(f"Epochs: {epoch + 1}, train_loss: {train_loss_avg}, train_time: {timer.t()}, learning_rate: {learning_rate}", file=logf, flush=True)
            self.ckp.train_log[-1] = train_loss.item()
            print("epoch: ", epoch + 1, "loss: ", train_loss_avg, file=logf, flush=True)
            print(file=logf, flush=True)

        self.scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()
        logf.close()
    
    def eval(self) -> None:
        logf: io.TextIOWrapper = open('/workspace/logs/valLog.txt', 'a+')

        torch.cuda.empty_cache()
        gc.collect()

        if not self.test_only:
            epoch = self.scheduler.last_epoch
        elif self.test_only or self.testing_spec["save_img"]: # test_only
            epoch = 0

        outputs = []
        metrics = self.testing_spec['metrics']
        if epoch % self.testing_spec["test_every"] == 0:
            self.ckp.add_val_log(torch.zeros([1, len(metrics)]))
            
            timer = utils.Timer()
            rmse: Tensor = torch.zeros(1, device=self.gpu_id)
            psnr: Tensor = torch.zeros(1, device=self.gpu_id)

            total_imgs: Tensor = torch.zeros(1, device=self.gpu_id)
            recon_img = None

            with torch.inference_mode():
                for iters, (img_lq, img_gt, file_index) in enumerate(tqdm(self.test_loader, disable=(self.gpu_id != 0))):
                    # for batch, (sino, img, loc_name) in enumerate(tqdm(self.loader_test, disable=(self.gpu_id != 0))):
                    img_lq, img_gt = img_lq.cuda(self.gpu_id), img_gt.cuda(self.gpu_id)

                    # Inference the batch size
                    batch_size = img_lq.shape[0]
                    total_imgs += batch_size

                    # img_lq = utils.normalize(img_lq, -283, 441)
                    print(f"Processing video index: {file_index.item()}", file=logf, flush=True)

                    with torch.autocast(device_type="cuda"):
                        if self.testing_spec["crop"]:
                            # use for test video clipping(with padding for swin transformers)
                            if not self.test_only:
                                self.testing_spec["tile_overlap"] = [0, 10, 0]
                            
                            if self.model_type == "KernelWizard":
                                recon_img: Tensor = utils.test_video(img_lq, self.model, self.gpu_id, self.testing_spec, img_gt)
                            else:
                                recon_img: Tensor = utils.test_video(img_lq, self.model, self.gpu_id, self.testing_spec)
                        else:
                            recon_img: Tensor = self.model(img_lq)
                    
                    if self.testing_spec["save_img"] or self.test_only:
                        print("A shape of reconstructed image tensor:", recon_img.squeeze(0).shape)
                        outputs.append(recon_img.squeeze(0).half()) # To address shape mismatch error

                        # Plot along the depth dimension
                        for i in range(recon_img.shape[2]):
                            plt.figure(figsize=(15, 5))
                            plt.title(f"{i+1}-th frame at the Reconstructed video {file_index.item() + 1}")
                            plt.imshow(recon_img[:, :, i, ...].squeeze(0).cpu().permute(1, 2, 0).numpy())
                            plt.axis('off')
                            os.makedirs(f'/workspace/figures/Reconstructed_video_{file_index.item() + 1}', exist_ok=True)
                            plt.savefig(fname=f'/workspace/figures/Reconstructed_video_{file_index.item() + 1}/frame_{i+1}.png')
                            plt.close()                        

                    match self.model_type:
                        case "ReBotNet":
                            img_gt: Tensor = img_gt[:, :, 1:, ...]
                        case "EDVR":
                            img_gt: Tensor = img_gt[:, :, 2:-2, ...]
                        case "KernelWizard":
                            img_gt: Tensor = img_lq  # recon_img is not sharp. It is blur.
                        case "MIMOUNet":
                            recon_img: Tensor = recon_img[-1]
                        case _:
                            raise ValueError(f"Unknown model type: {self.model_type}")

                    if "RMSE" in metrics:
                        rmse += utils.calc_rmse_per_slice(recon_img.cpu(), img_gt.clamp_(0, 1).cpu())*batch_size
                        print(f'Epoch: {epoch}, Iter: {iters + 1}, RMSE: {rmse.item()/total_imgs.item()}', file=logf, flush=True)
                    if "PSNR" in metrics:
                        psnr += utils.calc_psnr_per_slice(recon_img.cpu(), img_gt.clamp_(0, 1).cpu())*batch_size
                        print(f'Epoch: {epoch}, Iter: {iters + 1}, PSNR: {psnr.item()/total_imgs.item()}', file=logf, flush=True)
                    print(file=logf, flush=True)

                    del img_lq, img_gt, recon_img
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.testing_spec["save_img"] or self.test_only:
                    all_outputs: Tensor = torch.cat(outputs, dim=1).cpu().half()
                
                # Sum scalar tensors across all processors.
                if self.ddp:
                    dist.all_reduce(rmse, op=dist.ReduceOp.SUM)
                    dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_imgs, op=dist.ReduceOp.SUM)

                    # gather outputs from all GPUs
                    if self.testing_spec["save_img"] or self.test_only:
                        gathered_list: list[Tensor] = [torch.zeros_like(all_outputs, dtype=torch.float16, device='cpu') for _ in range(self.opt["num_gpu"])]
                        dist.all_gather(gathered_list, all_outputs)
                        all_outputs = torch.cat(gathered_list, dim=0)
                        
                        del gathered_list
                        torch.cuda.empty_cache()
                        gc.collect()
        
                if self.test_only:
                    torch.save(all_outputs, self.ckp.get_path('results/test_outputs.pt'))

                elif self.testing_spec["save_img"]:
                    torch.save(all_outputs, self.ckp.get_path(f"results/val_outputs_{epoch//self.testing_spec['test_every']-1}.pt"))
                
                if self.testing_spec["save_img"] or self.test_only:
                    del all_outputs, outputs
                torch.cuda.empty_cache()
                gc.collect()

                if self.gpu_id == 0:
                    self.ckp.val_log[-1, 0] = rmse / total_imgs
                    # best[0] is the minimum value, best[1] is the index of the minimum value
                    best = self.ckp.val_log[:, 0].min(0)
                    self.ckp.val_log[-1, 1] = psnr / total_imgs

                    print(f'Epochs: {epoch}, val_rmse: {self.ckp.val_log[-1, 0].item()}, val_psnr: {self.ckp.val_log[-1, 1].item()}, val_time: {timer.t()}', file=logf, flush=True)

                    if not self.test_only:
                    # 가장 rmse가 작은 모델의 index가 현재 epoch//test_every와 같다면 best model 저장
                        self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch // self.testing_spec["test_every"]), n_gpus=self.opt["num_gpu"])
                        print("Model Saved at Epoch: ", epoch, "RMSE: ", self.ckp.val_log[-1, 0].item(), "PSNR: ", self.ckp.val_log[-1, 1].item(), ", Best RMSE: ", best[0].item(), file=logf, flush=True)
                        print(file=logf, flush=True)
                    else:
                        torch.save(self.ckp.val_log, self.ckp.get_path("test_rmse_log.pt"))
                        print(f"RMSE: {self.ckp.val_log[-1, 0].item()}, PSNR: {self.ckp.val_log[-1, 1].item()}", file=logf, flush=True)
                        print(file=logf, flush=True)

        torch.cuda.empty_cache()
        gc.collect()
        logf.close()

    # only used for training
    def terminate(self) -> bool:
        epoch: int = self.scheduler.last_epoch
        return epoch >= self.training_spec["details"]['epochs']
