import torch
import random
import numpy as np
from typing import Any
from torch import Tensor

def extract_tubelets(video_tensor: Tensor, temporal_size: int=16, spatial_size: int=128) -> Tensor:
    """
    Extract tubelets of shape (3, 16, 128, 128) from input (3, N, 768, 1280).
    Ensures deterministic indexing-based slicing.
    """
    C, N, H, W = video_tensor.shape  # (3, N, 720, 1280)
    
    # Ensure N is a multiple of 16 for exact tubelet extraction
    T_slices: int = N // temporal_size
    H_slices: int = H // spatial_size  # 768 / 128 = 6
    W_slices: int = W // spatial_size  # 1280 / 128 = 10

    tubelets = []
    for t in range(T_slices):
        for h in range(H_slices):
            for w in range(W_slices):
                tubelet: Tensor = video_tensor[
                    :,  # Keep channels
                    t * temporal_size : (t + 1) * temporal_size,  # Temporal slice
                    h * spatial_size : (h + 1) * spatial_size,  # Height slice
                    w * spatial_size : (w + 1) * spatial_size,  # Width slice
                ]
                tubelets.append(tubelet)

    # Stack tubelets into a single tensor
    return torch.stack(tubelets)  # Shape: (num_tubelets, 3, 16, 128, 128)

def reconstruct_video(tubelets: Tensor, N: int, H: int=720, W: int=1280, spatial_size: int=128) -> Tensor:
    """
    Reconstruct the padded (3, N, 768, 1280) tensor from tubelets.
    """
    C, temporal_size, _, _ = tubelets.shape[1:]
    
    # Create empty tensor for reconstruction
    reconstructed_video: Tensor = torch.zeros(C, N, H, W)

    # Number of tiles in H and W
    H_slices: int = H // spatial_size
    W_slices: int = W // spatial_size
    T_slices: int = N // temporal_size

    idx = 0
    for t in range(T_slices):
        for h in range(H_slices):
            for w in range(W_slices):
                reconstructed_video[
                    :, 
                    t * temporal_size : (t + 1) * temporal_size, 
                    h * spatial_size : (h + 1) * spatial_size, 
                    w * spatial_size : (w + 1) * spatial_size
                ] = tubelets[idx]
                idx += 1

    return reconstructed_video


def get_local_split(items: list, world_size: int, rank: int, seed: int) -> np.ndarray[Any, Any] | np.ndarray[Any, np.dtype[Any]]:
    """ The local rank only loads a split of the dataset. """
    n_items: int = len(items)
    items_permute: np.ndarray = np.random.RandomState(seed).permutation(items)
    if n_items % world_size == 0:
        padded_items: np.ndarray = items_permute
    else:
        padding: np.ndarray[Any, Any] = np.random.RandomState(seed).choice(
            items,
            world_size - (n_items % world_size),
            replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank: int = len(padded_items) // world_size
    local_items: np.ndarray[Any, Any] = padded_items[n_per_rank * rank: n_per_rank * (rank+1)]

    return local_items

def N_dim_crop(data_dim: int, lq_tensor: torch.Tensor, gt_tensor: torch.Tensor, patch_size: int, frame_crop_size: int = None) -> tuple[Tensor, Tensor]:
    """ RandomCrop the input and GT tensor
    Args:
        data_dim: int, the dimension of the data
        lq_tensor: torch.Tensor, the low quality tensor
        gt_tensor: torch.Tensor, the ground truth tensor
        patch_size: int, the size of the patch
       
        *args: additional cropping in 3D or 4D data
        frame_crop_size: int, the size of the frame crop

    Returns:
        img_lq: torch.Tensor, the cropped low quality tensor
        img_gt: torch.Tensor, the cropped ground truth tensor
    """

    if data_dim == 2:
        c, h, w = lq_tensor.shape
        crop_h: int = random.randint(0, h-patch_size)
        crop_w: int = random.randint(0, w-patch_size)
        img_lq: Tensor = lq_tensor[:, crop_h:crop_h+patch_size, crop_w:crop_w+patch_size]
        img_gt: Tensor = gt_tensor[:, crop_h:crop_h+patch_size, crop_w:crop_w+patch_size]

    elif data_dim == 3:
        c, t, h, w = lq_tensor.shape
        crop_t: int = random.randint(0, t-frame_crop_size)
        crop_h: int = random.randint(0, h-patch_size)
        crop_w: int = random.randint(0, w-patch_size)
        img_lq: Tensor = lq_tensor[:, crop_t:crop_t+frame_crop_size, crop_h:crop_h+patch_size, crop_w:crop_w+patch_size]
        img_gt: Tensor = gt_tensor[:, crop_t:crop_t+frame_crop_size, crop_h:crop_h+patch_size, crop_w:crop_w+patch_size]

    else:
        raise ValueError(f'Invalid data_dim: {data_dim}')

    return img_lq, img_gt


def augment(data_dim: int, lq_tensor: torch.Tensor, gt_tensor: torch.Tensor, flip: bool = True, rotate: bool = True) -> tuple[Tensor, Tensor]:
    """ Augment the data
    Args:
        lq_tensor: torch.Tensor, the data to augment, c, t, h, w
        gt_tensor: torch.Tensor, the data to augment, c, t, h, w
        flip: bool, whether to flip the data
        rotate: bool, whether to rotate the data
    Returns:
        augmented_data: torch.Tensor, the augmented data
    """

    added_dim: int = data_dim - 1

    if flip:
        if random.random() > 0.5:
            lq_tensor = torch.flip(lq_tensor, [0+added_dim])
            gt_tensor = torch.flip(gt_tensor, [0+added_dim])
        if random.random() > 0.5:
            lq_tensor = torch.flip(lq_tensor, [1+added_dim])
            gt_tensor = torch.flip(gt_tensor, [1+added_dim])
    if rotate:
        k = random.randint(0, 3)
        lq_tensor = torch.rot90(lq_tensor, k, [0+added_dim, 1+added_dim])
        gt_tensor = torch.rot90(gt_tensor, k, [0+added_dim, 1+added_dim])
    
    print(f'Augmented blur tensor shape: {lq_tensor.shape}')
    print(f'Augmented sharp tensor shape: {gt_tensor.shape}')
    return lq_tensor, gt_tensor
