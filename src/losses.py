import torch
from torch import nn, Tensor
import torch.nn.functional as F
from monai.losses.ssim_loss import SSIMLoss

class CharbonnierLoss(nn.Module):
    def __init__(self, sigma: float = 0.01) -> None:
        """
        Charbonnier Loss (a differentiable variant of L1 loss).

        Args:
            sigma (float, optional): A small constant for numerical stability. Defaults to 0.01.
        """
        super(CharbonnierLoss, self).__init__()
        self.sigma = sigma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute the Charbonnier loss between two tensors.

        Args:
            x (Tensor): Predicted tensor.
            y (Tensor): Ground truth tensor.

        Returns:
            Tensor: Scalar loss value.
        """
        x = torch.sqrt((y - x) ** 2 + self.sigma ** 2)
        return x.mean()

class CharbonnierSSIMLoss(nn.Module):
    def __init__(self, spatial_dims: int = 2, lamda: float = 1.0, sigma: float = 0.01) -> None:
        """
        Combined Charbonnier + SSIM Loss.

        Args:
            spatial_dims (int, optional): Number of spatial dimensions (2 for images, 3 for volumes). Defaults to 2.
            lamda (float, optional): Weighting factor for the SSIM loss component. Defaults to 1.0.
            sigma (float, optional): Stability constant for Charbonnier loss. Defaults to 0.01.
        """
        super(CharbonnierSSIMLoss, self).__init__()
        self.ssim = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0)
        self.lamda: float = lamda
        self.charbonnier = CharbonnierLoss(sigma=sigma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute the combined Charbonnier and SSIM loss.

        Args:
            x (Tensor): Predicted tensor.
            y (Tensor): Ground truth tensor.

        Returns:
            Tensor: Scalar loss value.
        """
        ssim: Tensor = self.ssim(x, y)
        charbonnier: Tensor = self.charbonnier(x, y)
        loss: Tensor = ssim * self.lamda + charbonnier
        return loss.mean()


class MSFRLoss(nn.Module):
    def __init__(self, content_loss: nn.Module = nn.L1Loss(), lamda: float = 0.1, scales: int = 3, 
                 scale_weights: list[float] = None) -> None:
        """
        Multi-Scale Frequency Reconstruction Loss from
        'Rethinking Coarse-to-Fine Approach in Single Image Deblurring' (ICCV 2021).

        This combines content loss (L1) and frequency loss (L1 between FFT magnitudes)
        at multiple image scales.

        Args:
            content_loss (nn.Module): Base pixel-wise loss (e.g., nn.L1Loss or nn.MSELoss).
            lamda (float): Weight factor for frequency-domain loss.
            scales (int): Number of multi-scale levels (starting from coarsest).
            scale_weights (list[float], optional): Optional per-scale weighting.
        """
        super(MSFRLoss, self).__init__()

        self.content_loss: nn.Module = content_loss
        self.lamda: float = lamda
        self.scales: int = scales
        self.scale_weights: list[float] = scale_weights or [1.0 for _ in range(scales)]
        assert len(self.scale_weights) == self.scales

    def fft_l1_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_fft: Tensor = torch.fft.fft2(pred, norm='ortho')
        target_fft: Tensor = torch.fft.fft2(target, norm='ortho')
        pred_mag: Tensor = torch.abs(pred_fft)
        target_mag: Tensor = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)

    def forward(self, pred: list[Tensor], target: Tensor) -> Tensor:
        """
        Compute MSFR loss.

        Args:
            pred (list[Tensor]): List of prediction tensors [coarse → fine scale].
            target (Tensor): Ground-truth high-resolution tensor.

        Returns:
            Tensor: Total scalar loss value.
        """
        loss_content = 0.0
        loss_freq = 0.0
        gt_scaled: Tensor = target

        for k in range(self.scales-1, 0, -1):
            if k < self.scales - 1:
                gt_scaled: Tensor = F.interpolate(gt_scaled, scale_factor=0.5, mode='bilinear', align_corners=False)
                
            loss_content += self.scale_weights[k] * self.content_loss(pred[k], gt_scaled)
            loss_freq += self.scale_weights[k] * self.fft_l1_loss(pred[k], gt_scaled)


        return loss_content + self.lamda * loss_freq
