import io
import torch
from typing import Any
from torch import nn
from typing import Union
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

#####################################################################################################
################################## Base U-net Modules ###############################################
#####################################################################################################

class SpatioBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int)->None:
        super(SpatioBlock, self).__init__()
        self.input = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding='same', padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding='same', padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(in_channels),]
        )
        self.output = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding='same', padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x1 = self.input[1](self.input[0](x))
        x2 = self.input[2](x1.permute(0, 2, 3, 4, 1))
        x3 = x2.permute(0, 4, 1, 2, 3)

        x4 = self.input[4](self.input[3](x3))
        x5 = self.input[5](x4.permute(0, 2, 3, 4, 1))
        x6 = x5.permute(0, 4, 1, 2, 3)

        return self.output(x6 + x)

class SpatioTemporalBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int)->None:
        super(SpatioTemporalBlock, self).__init__()

        self.upsampling = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.input = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(in_channels),]
        )
        self.output = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x1 = self.upsampling(x)

        x2 = self.input[1](self.input[0](x1))
        x3 = self.input[2](x2.permute(0, 2, 3, 4, 1))
        x4 = x3.permute(0, 4, 1, 2, 3)

        x5 = self.input[4](self.input[3](x4))
        x6 = self.input[5](x5.permute(0, 2, 3, 4, 1))
        x7 = x6.permute(0, 4, 1, 2, 3)

        return self.output(x7 + x1)

###############################################################################################################
################################## Temporal Transformer Modules ###############################################
###############################################################################################################

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, num_patches: int) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.projection = nn.Sequential(nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
                                        nn.GELU(),
                                        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding='same', padding_mode='replicate'),
                                        nn.GELU(),
                                        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding='same', padding_mode='replicate'),
                                        nn.GELU())
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Nc, H, W = x.shape  # Batch, Channels, Temporal, Height, Width
        x = x.permute(0, 2, 1, 3, 4)  # (B, Nc, C, H, W)
        x = x.reshape(B * Nc, C, H, W)  # Merge batch and time dims
        x = self.projection(x)  # Patch embedding
        x = x.flatten(2).transpose(1, 2)  # Flatten spatial dims
        x = x + self.pos_embedding  # Add positional embedding
        return x.reshape(B, Nc, -1, x.shape[-1])  # Reshape back to include time dimension

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU())
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Nc, Patches, D)
        B, Nc, P, D = x.shape
        x = x.reshape(B * P, Nc, D)  # Reshape for attention
        x = self.layer_norm1(x)
        attn_out, _ = self.attn1(x, x, x)
        mlp_out = self.mlp1(attn_out + x)
        mlp_out = mlp_out + attn_out
        x = self.layer_norm2(mlp_out)
        attn_out, _ = self.attn2(x, x, x)
        mlp_out = self.mlp2(attn_out + mlp_out)
        x = mlp_out + attn_out
        return x.reshape(B, P, Nc, D).permute(0, 2, 1, 3)  # Reshape back

class PatchUnembedding(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_proj = nn.Sequential(nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size),
                                      nn.GELU(),
                                      nn.Conv2d(out_channels, out_channels*4, kernel_size=3, padding='same', padding_mode='replicate'),
                                      nn.GELU(),
                                      nn.Conv2d(out_channels*4, out_channels, kernel_size=3, padding='same', padding_mode='replicate'),
                                      nn.GELU(),)
    
    def forward(self, x: torch.Tensor, orig_size: int) -> torch.Tensor:
        B, Nc, P, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B * Nc, P, D)  # Merge batch & time dims
        H, W = orig_size // self.patch_size, orig_size // self.patch_size
        x = x.transpose(1, 2).reshape(B * Nc, D, H, W)  # Reshape
        x = self.out_proj(x)  # Upsample
        x = x.reshape(B, Nc, -1, orig_size, orig_size)  # Reshape back to (B, Nc, out_channels, H, W)
        return x.permute(0, 2, 1, 3, 4)  # (B, out_channels, Nc, H, W)

class TemporalTransformer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, patch_size: int, img_size: int, num_heads: int) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size, num_patches)
        self.temporal_attention = TemporalSelfAttention(embed_dim, num_heads)
        self.patch_unembedding = PatchUnembedding(embed_dim, out_channels, patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-1]  # Preserve original size
        x = self.patch_embedding(x)
        x = self.temporal_attention(x)
        x = self.patch_unembedding(x, orig_size)
        return x  # Output shape (B, out_channels, Nc, H, W)

##############################################################################################################
######################################## Main Modules ########################################################
##############################################################################################################

class TTUNET(nn.Module):
    """
    The main experiment module.
    """
    def __init__(self, in_channels:int, out_channels:int, debug:bool, checkpoint:bool)->None:
        super(TTUNET, self).__init__()

        self.checkpoint: bool = checkpoint
        self.debug: bool = debug

        self.spatioconv1 = SpatioBlock(in_channels, 8)
        
        self.spatioconv2 = SpatioBlock(8, 16)
        
        self.spatioconv3 = SpatioBlock(16, 32)

        self.spatioconv4 = SpatioBlock(32, 64)
        
        self.TT1 = TemporalTransformer(64, 64, embed_dim=128, patch_size=4, img_size=16, num_heads=4)
        self.bottle_norm = nn.LayerNorm(64)

        self.TT2 = TemporalTransformer(64, 64, embed_dim=128, patch_size=4, img_size=32, num_heads=4)
        self.spatio_temporal_conv1 = SpatioTemporalBlock(64+64, 64)
        self.layer_norm1 = nn.LayerNorm(64)
        
        self.TT3 = TemporalTransformer(32, 32, embed_dim=64, patch_size=4, img_size=64, num_heads=2)
        self.spatio_temporal_conv2 = SpatioTemporalBlock(64+32, 32)
        self.layer_norm2 = nn.LayerNorm(32)

        self.TT4 = TemporalTransformer(16, 16, embed_dim=32, patch_size=4, img_size=128, num_heads=1)
        self.spatio_temporal_conv3 = SpatioTemporalBlock(32+16, 16)
        self.layer_norm3 = nn.LayerNorm(16)

        self.TT5 = TemporalTransformer(8, 8, embed_dim=16, patch_size=4, img_size=256, num_heads=1)
        self.spatio_temporal_conv4 = SpatioTemporalBlock(16+8, 8)
        self.layer_norm4 = nn.LayerNorm(8)
        
        self.output = nn.Conv3d(8, out_channels, kernel_size=1)
        
        self.maxpool = nn.MaxPool3d(2)

    def copy_and_crop(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        # Calculate the starting indices for cropping
        start_indices: list[int] = [(y.size(dim) - x.size(dim)) // 2 for dim in range(2, y.dim())]

        # Crop y to match the shape of x
        slices: list[slice[int | Any, int | Any, int | Any]] = [slice(start, start + x.size(dim)) for start, dim in zip(start_indices, range(2, y.dim()))]

        return y[(slice(None), slice(None)) + tuple(slices)]

    def _forward(self, x:torch.Tensor)->torch.Tensor:
        if self.debug:

            logf: io.TextIOWrapper = open(file='/workspace/logs/TTUNETLog.txt', mode='a+')
            print("\n[DEBUG] **Input Tensor Stats**", file=logf, flush=True)
            print(f"  - Max input: {x.max()}, Min input: {x.min()}", file=logf, flush=True)
            print(f"  - NaN in input: {x.isnan().any()}, Inf in input: {x.isinf().any()}", file=logf, flush=True)

            # Encoder Path
            x1 = self.spatioconv1(x)
            x2 = self.maxpool(x1)

            x3 = self.spatioconv2(x2)
            x4 = self.maxpool(x3)

            x5 = self.spatioconv3(x4)
            x6 = self.maxpool(x5)

            x7 = self.spatioconv4(x6)
            x8 = self.maxpool(x7)

            x9 = self.TT1(x8)
            x9 = self.bottle_norm(x9.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            # Decoder Path
            residual = self.copy_and_crop(x=x9, y=self.TT2(x7) + x7)
            x10 = self.spatio_temporal_conv1(torch.cat([x9, residual], 1))
            x10 = self.layer_norm1(x10.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            residual = self.copy_and_crop(x=x10, y=self.TT3(x5) + x5)
            x11 = self.spatio_temporal_conv2(torch.cat([x10, residual], 1))
            x11 = self.layer_norm2(x11.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            residual = self.copy_and_crop(x=x11, y=self.TT4(x3) + x3)
            x12 = self.spatio_temporal_conv3(torch.cat([x11, residual], 1))
            x12 = self.layer_norm3(x12.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            residual = self.copy_and_crop(x=x12, y=self.TT5(x1) + x1)
            x13 = self.spatio_temporal_conv4(torch.cat([x12, residual], 1))
            x13 = self.layer_norm4(x13.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            output = self.output(x13) + x
            
            # Debugging Visualization
            stages = {
                "input": x,
                "spatioconv1": x1, "maxpool1": x2,
                "spatioconv2": x3, "maxpool2": x4,
                "spatioconv3": x5, "maxpool3": x6,
                "spatioconv4": x7, "maxpool4": x8,
                "TemporalTransformer": x9,
                "spatio_temporal_conv1": x10, "spatio_temporal_conv2": x11, "spatio_temporal_conv3": x12, "spatio_temporal_conv4": x13,
                "final_output": output
            }

            for name, tensor in stages.items():
                if name == "input" or name == "final_output":
                    print(f"[DEBUG] After {name}: Max {tensor.max()}, Min {tensor.min()}", file=logf, flush=True)
                    plt.figure(figsize=(10, 10))
                    plt.title(f'TTUNET {name}')
                    patch = tensor[0].clone().detach().cpu()
                    plt.imshow(torch.clamp(patch.to(dtype=torch.float32).permute(1, 2, 3, 0)[0], min=0, max=1).numpy())
                    plt.axis('off')
                    plt.colorbar()
                    plt.savefig(f'/workspace/figures/model_{name}.png')
                    plt.close()
                else:
                    print(f"[DEBUG] After {name}: Max {tensor.max()}, Min {tensor.min()}", file=logf, flush=True)
                    plt.figure(figsize=(10, 10))
                    plt.title(f'TTUNET {name}')
                    patch = tensor[0].clone().detach().cpu()
                    plt.imshow(torch.clamp(patch.to(dtype=torch.float32).permute(1, 2, 3, 0)[0].mean(2), min=0, max=1).numpy())
                    plt.axis('off')
                    plt.colorbar()
                    plt.savefig(f'/workspace/figures/model_{name}.png')
                    plt.close()
            logf.close()

            return output

        elif not self.debug:        
            x1 = self.spatioconv1(x)
            x2 = self.maxpool(x1)
        
            x3 = self.spatioconv2(x2)
            x4 = self.maxpool(x3)     

            x5 = self.spatioconv3(x4)
            x6 = self.maxpool(x5)
            
            x7 = self.spatioconv4(x6)
            x8 = self.maxpool(x7)

            x9 = self.TT1(x8)
            x9 = self.bottle_norm(x9.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

            # Decoder Path
            residual = self.copy_and_crop(x=x9, y=self.TT2(x7))
            x10 = self.spatio_temporal_conv1(torch.cat([x9, residual], 1))
            x10 = self.layer_norm1(x10.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            residual = self.copy_and_crop(x=x10, y=self.TT3(x5))
            x11 = self.spatio_temporal_conv2(torch.cat([x10, residual], 1))
            x11 = self.layer_norm2(x11.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            residual = self.copy_and_crop(x=x11, y=self.TT4(x3))
            x12 = self.spatio_temporal_conv3(torch.cat([x11, residual], 1))
            x12 = self.layer_norm3(x12.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            
            residual = self.copy_and_crop(x=x12, y=self.TT5(x1))
            x13 = self.spatio_temporal_conv4(torch.cat([x12, residual], 1))
            x13 = self.layer_norm4(x13.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            output = self.output(x13) + x

            return output
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        elif not self.checkpoint:
            return self._forward(x)

if __name__ == '__main__':
    import torch
    import warnings

    gpu_ok = False
    if torch.cuda.is_available():
        device_cap: torch.Tuple[int] = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True

    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )

    # Test SpatioBlock
    CBlcok = SpatioBlock(3, 3)
    summary(CBlcok, input_size=(1, 3, 16, 64, 64))
    # Test SpatioTemporalBlock
    EBlock = SpatioTemporalBlock(3, 3)
    summary(EBlock, input_size=(1, 3, 16, 64, 64))
    # Test TemporalTransformer
    TT = TemporalTransformer(512, 512, embed_dim=100, patch_size=2, img_size=16, num_heads=4)
    summary(TT, input_size=(1, 512, 2, 16, 16))
    # Test TTUNET
    dt: torch.dtype = torch.float32
    model = TTUNET(3, 3, debug=False, checkpoint=False).to(dtype=dt)
    print("total number of parameter is {}".format(sum(p.numel() for p in model.parameters())))
    summary(TTUNET(3, 3, debug=False, checkpoint=False).to(dtype=dt), input_size=(4, 3, 16, 256, 256), dtypes=[dt])
