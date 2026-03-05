# Benchmark-Study-on-Video-Restoration-Models
Conducted a comparative analysis of 5 different motion deblurring models using the GoPro dataset. Focused on the trade-off between restoration quality (PSNR) and inference efficiency.

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/6e0ff0ea-9cbe-4b60-b304-4624fd870265" />

## Core Metric comparison
<img width="1626" height="813" alt="image" src="https://github.com/user-attachments/assets/91e730ff-bd00-438e-8719-c9fb55a2e4a8" />

## Inference Speed per batch
<img width="1150" height="737" alt="image" src="https://github.com/user-attachments/assets/6b553cd0-a8a7-4c68-979d-64fa7db80fc8" />
- Batch sizes were adjusted to fully utilize the 12GB VRAM of the Titan Xp
- UNet-3D is not shown in this figure because there is no recorded latency.

## Efficiency comparison
<img width="1051" height="841" alt="image" src="https://github.com/user-attachments/assets/786a9a7f-a73e-4159-8e45-4998b5b70007" />
- MIMO-U-Net offered a good trade-off between latency and restoration quality.
- ReBotNet showed promising efficiency in latency but suffered from training instability and suboptimal quality.
- EDVR demonstrated the most stable performance among the tested. 
- Since TT-U-Net was originally designed for CT video artifact reduction, it performed poorly on the motion deblurring task.
- UNet-3D showed the lowest quality and is not shown in this figure because there is no recorded latency.

## Summary table
<img width="1888" height="791" alt="image" src="https://github.com/user-attachments/assets/83cf9452-de73-420a-91db-177b6cd55f31" />



## 1. General Experimental Setup
All experiments were conducted under the following common environmental conditions:
- Hardware: NVIDIA Titan Xp GPU (4-GPU configuration)
- Dataset: GOPRO_Large dataset (Standard Train/Test split)
- Framework & Libraries: PyTorch (with specific dependencies such as mmcv for deformable convolution)
<img width="1641" height="360" alt="image" src="https://github.com/user-attachments/assets/8c85b07c-370a-4943-98d2-02618fba1565" />

## EDVR (Video Restoration with Enhanced Deformable Convolutional Networks)
<img width="1376" height="458" alt="image" src="https://github.com/user-attachments/assets/4262c3ff-6631-48e0-808f-e9aaef5e9a34" />

### Training Configuration:
- Input: 5 consecutive frames (Output: Center frame).
- Hyperparameters: Batch Size 16, Patch Size 256.
- Optimizer: AdamW with an initial Learning Rate (LR) of 1e-4.
- Scheduler: MultiStepLR (Milestones: 120, 180, 240, 270 epochs; Gamma: 0.5).
- Loss Function: Charbonnier Loss

### Outcome:
It achieved the most reliable performance among the deblurring models.
- Best Validation PSNR: 29.40 dB (at Epoch 1000).
- Best RMSE: 0.0361.

### Challenges & Key Findings
- Dependency Conflict Resolution:
  - Encountered compatibility issues due to deprecated Deformable Convolution modules in the original code.
  - Successfully resolved by identifying and integrating the specific version of the mmcv library required for execution.
- Performance Analysis:
  - Achieved the highest stability and performance among the experimental group with 29.40 dB PSNR.
  - Observed smooth loss convergence, attributed to the effectiveness of the Charbonnier Loss.
- Limitations:
  - Performance degradation was observed in specific blur scenarios, particularly with Defocus Deblurring.

## ReBotNet (Recurrent Bottleneck Network)
<img width="1496" height="499" alt="image" src="https://github.com/user-attachments/assets/2d5abb55-504e-4c32-9ece-5606d6c42712" />

### Training Configuration:
- Input: Consecutive 2 frames.
- Hyperparameters: Batch Size 2, Crop Size 384.
- Optimizer: AdamW with an initial LR of 1e-4.
- Scheduler: MultiStepLR (Same milestones as EDVR).
- Loss Function: CharbonnierLoss

### Outcome:
- Best PSNR: ~26.77 dB.
- It failed to reach the target performance (30dB) and showed lower quality compared to EDVR and MIMO-UNet.

### Challenges & Key findings
<img width="950" height="314" alt="image" src="https://github.com/user-attachments/assets/99a0d6f9-df26-4ac9-b91f-45ea9ed7a786" />

- Critical Parameter Discrepancy: 
  - Identified a significant inconsistency between the paper's claim (~3.2M parameters) and the actual implementation code (~133M parameters). 
  - Attempted to clarify this issue by contacting the author via GitHub repository, but received no response. 
- Training Instability: 
  - Although the model showed promising performance metrics, the validation performance exhibited severe oscillation as training progressed, raising concerns about reliability. 
- Dimension Errors: 
  - Encountered frequent dimension-related errors during execution due to the specific architecture of processing 2 input patches to generate 1 output frame, requiring careful handling. 

## MIMO-U-Net (Multi-Input Multi-Output U-Net)

### Traing Configuration:
- Input: Single image (Multi-scale).
- Hyperparameters: Batch Size 4, Crop Size 256.
- Optimizer: AdamW with an initial LR of 1e-4.
- Scheduler: MultiStepLR.
- Loss Function: MSFRLoss (Multi-Scale Frequency Reconstruction Loss) combined with L1 Loss. This loss focuses on restoring frequency details.

### Outcome:
- Best PSNR: Approximately 30.6 dB (based on the metrics graph).
- It outperformed ReBotNet, positioning itself as a strong candidate for practical deployment.

### Key Findings

https://github.com/user-attachments/assets/af45c1be-cd2e-46d1-88e1-0f1b8021cfba

Balanced Trade-off:
- Evaluated as the model achieving the most optimal balance between restoration quality (PSNR) and inference latency, making it a strong candidate for practical deployment.
Training Stability:
- Demonstrated consistent and stable convergence throughout the 1000-epoch training process, confirming its robustness.



## TT-U-Net (Temporal Transformer U-Net)

<img width="1335" height="445" alt="image" src="https://github.com/user-attachments/assets/18a25eb0-fe75-4c5d-a7fa-a481b42eca40" />


### Training Configuration:
- Input: 16 frames (Large temporal window).
- Hyperparameters: Batch Size 4, Crop Size 256.
- Optimizer: AdamW with an initial LR of 1e-4.
- Scheduler: CosineAnnealingLR (T_max: 100, Minimum LR: 1e-6).
- Note: Unlike other models using Step LR, this used Cosine Annealing, periodically resetting the learning rate.
- Loss Function: L1Loss.

### Outcome:
- PSNR: 29.40 dB
- RMSE: 0.0361
- The performance was suboptimal for this specific task.

### Challenges & Key Findings
- Domain Mismatch: This model was originally designed for Cardiac CT artifact reduction.
- Performance: It struggled to adapt to the GoPro motion deblurring dataset, likely because the nature of "motion artifacts" in CT scans differs fundamentally from natural camera motion blur.


## UNet-3D (Custom)
<img width="1841" height="549" alt="image" src="https://github.com/user-attachments/assets/bc66be6f-4072-4d2f-a44c-65b38191448b" />

### Training Configuration:
- Architecture: Custom 3D Convolution-based original U-Net.
- Optimization: Due to excessive memory usage (OOM), the patch size was reduced (572x572 → 256x256), and the channel expansion in some blocks was lowered.

### Outcome:
- Best PSNR: 24.45 dB.
- SSIM: 0.776.
- This was the lowest performing model, proving that simply applying 3D convolutions without sophisticated alignment is ineffective for video deblurring.

### Challenges & Key Findings
**Resolved OOM Issues in 3D CNNs**: 
- Tackled extreme memory consumption in 3D convolution architectures by optimizing model components.
**Architectural Optimization**:
- Reduced memory footprint by reducing the patch sizes(572x572 => 256x256) and input block’s channel expansion(3 to 64 => 3 to 32).
- Conducted ablation studies on bottleneck structures and skip connection methods, discovering that parameter count (up to 88M) was not the primary factor for memory usage (maintained at ~2.6GB).
**Key Research Insights**:
- Conducted ablation studies revealing that memory costs in 3D Video Restoration are driven by activation map retention from multi-branch architectures (e.g., Gating/Skips), rather than parameter count.
- Concluded that explicit alignment modules (e.g., PCD in EDVR) are critical for video deblurring, as pure 3D convolutions face performance limitations.


