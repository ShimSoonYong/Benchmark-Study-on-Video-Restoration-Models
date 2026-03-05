import os
import cv2
import time
import glob
import torch
import random
import numpy as np
from torch import Tensor
from pathlib import Path
from data import data_utils
from os.path import dirname
import torch.utils.data as data
import torch.nn.functional as F


class GOPRO_Large(data.Dataset):
    def __init__(self, opt: dict, mode: str='train') -> None:
        self.mode: str = mode
        if mode == 'train':
            self.data_spec = opt['datasets']['train']
        elif mode == 'val':
            self.data_spec = opt['datasets']['val']
        elif mode == 'test':
            self.data_spec = opt['datasets']['test']
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
        self.data_dim = opt['data_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GT_paths, self.input_paths = self.__scan()
        # self.frame_size is required because all data's frame sizes(or lengths?) are different
        self.lq, self.gt, self.frame_sizes = self.__pair_and_load()
        print(f'Frame sizes: {self.frame_sizes}')
        
    # scan files within the input and GT directories
    def __scan(self) -> tuple[list, list]:
        GT_paths = []
        input_paths = []
        GT_paths += sorted(glob.glob(os.path.join(self.data_spec['gt_dir'], "*.png")))
        input_paths += sorted(glob.glob(os.path.join(self.data_spec['input_dir'], "*.png")))
        return GT_paths, input_paths

    def load_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def __pair_and_load(self) -> tuple[list, list, list]:
        lqs, gts, frame_sizes = [], [], []
        detector = None  # Tracks video name changes
        i = 0  # Tracks frame count within a video

        gt_stack = []  # Stores GT tensors per video
        lq_stack = []  # Stores LQ tensors per video

        for gt in self.GT_paths:
            video_name = dirname(dirname(gt))  # Extracts 'video_name'

            # Detect new video and store previous video frames
            if video_name != detector:
                if detector is not None:  # Avoid appending on first iteration
                    frame_sizes.append(i)
                    print(f'GT video depth: {len(gt_stack)}')
                    gts.append(torch.stack(gt_stack))  # Stack all images for previous video
                    lqs.append(torch.stack(lq_stack))  # Stack all images for previous video
                    gt_stack, lq_stack = [], []  # Reset for next video

                detector = video_name
                i = 0  # Reset frame count for new video

            i += 1  # Increment frame count

            # Find corresponding input image (ensuring exact match)
            paired_path = next((p for p in self.input_paths if video_name in p and os.path.basename(gt) in p), None)
            if paired_path is None:
                raise ValueError(f"No exact matching input path found for {gt}")

            print(f'Video: {video_name} | processing PNG file: {paired_path}')

            # Load images and store in stack
            gt_stack.append(self.load_image(gt))
            lq_stack.append(self.load_image(paired_path))

        # Append the final video batch
        frame_sizes.append(i)
        gts.append(torch.stack(gt_stack))  # Stack last video batch
        lqs.append(torch.stack(lq_stack))  # Stack last video batch

        return lqs, gts, frame_sizes



    def __len__(self) -> int:
        if self.mode == 'train':
            return len(self.lq)*self.data_spec['repeat']

        elif self.mode == 'val' or 'test':
            
            if self.data_dim == 2:
                return sum([size for size in self.frame_sizes])
            elif self.data_dim == 3:
                # Predicts 60 frames per one frame, and the number of frames per a video varies.
                # return int(sum([(frames // 16) for frames in self.frame_sizes]))
                return len(self.lq)
            else:
                raise ValueError(f'Invalid data_dim: {self.data_dim}')

        return len(self.gt)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'train':
            tensor_lq: torch.Tensor = self.lq[index // self.data_spec['repeat']]
            tensor_gt: torch.Tensor = self.gt[index // self.data_spec['repeat']]

            tensor_lq = tensor_lq.transpose(0, 1)  # Ensure proper shape
            tensor_gt = tensor_gt.transpose(0, 1)

            Channel, frame_size, Height, Width = tensor_lq.shape

            if self.data_dim == 2:
                rand_frame: int = random.randrange(0, frame_size)
                tensor_lq = tensor_lq[:, rand_frame, :, :]
                tensor_gt = tensor_gt[:, rand_frame, :, :]
                img_lq, img_gt = data_utils.N_dim_crop(
                    data_dim=self.data_dim, lq_tensor=tensor_lq, gt_tensor=tensor_gt, patch_size=self.data_spec['crop_size']
                )

            elif self.data_dim == 3:
                img_lq, img_gt = data_utils.N_dim_crop(
                    data_dim=self.data_dim, lq_tensor=tensor_lq, gt_tensor=tensor_gt,
                    patch_size=self.data_spec['crop_size'], frame_crop_size=self.data_spec['frame_crop_size']
                )

            else:
                raise ValueError(f'Invalid data_dim: {self.data_dim}')

            if self.data_spec['augment']:
                img_lq, img_gt = data_utils.augment(
                    data_dim=self.data_dim, lq_tensor=img_lq, gt_tensor=img_gt, flip=True, rotate=True
                )

            print(f'Blur Tensor shape: {img_lq.shape}')
            print(f'Sharp Tensor shape: {img_gt.shape}')
            return img_lq, img_gt

        elif self.mode in ['val', 'test']:

            if self.data_dim == 2:
                data_sum = 0
                file_index = None
                frame_index = None

                for i, frames in enumerate(self.frame_sizes):
                    data_sum += frames
                    if index < data_sum:
                        file_index: int = i
                        cur_index: int = index - (data_sum - frames)
                        frame_index = cur_index
                        break

                if file_index is None or frame_index is None:
                    raise IndexError(f"Index {index} out of range for dataset.")

                tensor_lq = self.lq[file_index].transpose(0, 1)[:, frame_index, :, :]
                tensor_gt = self.gt[file_index].transpose(0, 1)[:, frame_index, :, :]

            elif self.data_dim == 3:
                data_sum = 0
                file_index = index

                for i, frames in enumerate(self.frame_sizes):
                    data_sum += frames
                    if index < data_sum:
                        file_index: int = i
                        cur_index: int = index - (data_sum - (frames))
                        frame_index: int = cur_index
                        break

                tensor_lq = self.lq[index].transpose(0, 1)
                tensor_gt = self.gt[index].transpose(0, 1)
            else:
                raise ValueError(f'Invalid data_dim: {self.data_dim}')

            return tensor_lq, tensor_gt, file_index