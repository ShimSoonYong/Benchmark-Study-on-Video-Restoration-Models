import os
import io
import math
import torch
import warnings
import numpy as np
import torchvision
from functools import partial
from operator import mul, xor
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from typing import Union, Any, Callable
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim: int, drop_path: float=0., layer_scale_init_value: float=1e-6) -> None :
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, T, C, H, W) -> (N, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, T, H, W, C) -> (N, T, C, H, W)
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape: int, eps: float=1e-6, data_format: str="channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape: tuple[int] = (normalized_shape, )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape)==4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.shape)==5:
                x = self.weight[:,None, None, None] * x + self.bias[:, None, None, None]
            return x


def drop_path(x, drop_prob: float = 0., training: bool = False) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale: float, num_feat: int) -> None:

        class Transpose_Dim12(nn.Module):
            """ Transpose Dim1 and Dim2 of a tensor."""

            def __init__(self):
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                return x.transpose(1, 2)

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat*4, kernel_size=( 3, 3), padding=( 1, 1)))
                m.append(nn.PixelShuffle(2))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv2d(num_feat, num_feat, kernel_size=( 3, 3), padding=( 1, 1)))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat,  num_feat, kernel_size=( 3, 3), padding=( 1, 1)))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv2d(num_feat, num_feat, kernel_size=(3, 3), padding=( 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# helpers

def pair(t: Union[tuple, int]) -> tuple:
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x: Tensor, **kwargs) -> Tensor:

        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout :float=0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim: int, heads :int=8, dim_head :int=64, dropout :float=0.) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, tok_dim: int, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout :float=0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(tok_dim, FeedForward(tok_dim, mlp_dim, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))]))
    def forward(self, x: Tensor) -> Tensor:

        for tok_ff, ff in self.layers:

            x_tok = x.transpose(-1,-2)
            x = tok_ff(x_tok) + x_tok
            x = x.transpose(-1, -2)
            x = ff(x) + x
        return x


class ReBotNet(nn.Module):


    def __init__(self,
                 upscale: int=1,
                 in_channels: int=3,
                 temp_dim: int=2,
                 img_size: list[int]=[2, 128, 128],
                 debug: bool=False,
                 checkpoint: bool=False,
                 window_size: list[int]=[6, 8, 8],
                 depths: list[int]=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                 indep_reconsts: list[int]=[11, 12],
                 embed_dims: list[int]=[96, 192, 384, 768],
                 num_heads: list[int]=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                 mul_attn_ratio: float=0.75,
                 mlp_ratio: float=2.,
                 qkv_bias: bool=True,
                 qk_scale=None,
                 drop_path_rate: float=0.2,
                 norm_layer: nn.Module=nn.LayerNorm,
                 spynet_path: Union[None, str]=None,
                 pa_frames: int=2,
                 deformable_groups: int=16,
                 recal_all_flows: bool=False,
                 nonblind_denoising: bool=False,
                 use_checkpoint_attn: bool=False,
                 use_checkpoint_ffn: bool=False,
                 no_checkpoint_attn_blocks: list=[],
                 no_checkpoint_ffn_blocks: list=[],
                 mlp_dim: int=1024,
                 dropout: float=0.1,
                 bottle_dim: int=576,
                 bottle_depth: int=4,
                 patch_size: int=1,
                 layer_scale_init_value: float=1e-6, 
                 out_indices: list[int]=[0, 1, 2, 3],
                 dim_head: int=64
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising
        self.debug = debug
        self.checkpoint = checkpoint

        dims = embed_dims

        # conv_first
       
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels*img_size[0], dims[0], kernel_size=2, stride=2),#, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)#, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

        ### end ConvNext

        ### start bottleneck

        image_size = img_size[-1]//4
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        patch_dim = embed_dims[-1] * patch_height * patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        nn.Linear(patch_dim, embed_dims[-1]),
        )

        big_patch = 16
        patch_dim_big = 3 * big_patch * big_patch

        self.big_embedding1 = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = big_patch, p2 = big_patch),
        nn.Linear(patch_dim_big, embed_dims[-1]),
        )

        self.big_embedding2 = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = big_patch, p2 = big_patch),
        nn.Linear(patch_dim_big, embed_dims[-1]),
        )

        self.pool = nn.MaxPool1d(2, 2)

        self.bottleneck = Transformer(768, bottle_dim, bottle_depth, num_heads[-1], dim_head, mlp_dim, dropout)

        self.temporal_transformer = Transformer(768, bottle_dim, bottle_depth, num_heads[-1], dim_head, mlp_dim, dropout)

        # self.norm = norm_layer(embed_dims[-1])
        # self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction
        num_feat = embed_dims[-1]
        num_feat1 = embed_dims[-2]
        num_feat2 = embed_dims[-3]
        num_feat3 = embed_dims[-4]

        self.upsample1 = Upsample(2, num_feat)
        self.upsample2 = Upsample(2, num_feat)
        self.upsample3 = Upsample(2, num_feat1)
        self.upsample4 = Upsample(2, num_feat2)

        self.upsamplef1 = Upsample(2, num_feat3)
        # self.upsamplef2 = Upsample(4, num_feat3)

        self.conv_last = nn.Conv2d(num_feat3, 3, kernel_size=( 3, 3), padding=(1, 1))

        self.chchange1 = nn.Conv2d(num_feat, num_feat1, kernel_size=( 3, 3), padding=(1, 1))
        self.chchange2 = nn.Conv2d(num_feat1, num_feat2, kernel_size=( 3, 3), padding=(1, 1))
        self.chchange3 = nn.Conv2d(num_feat2, num_feat3, kernel_size=( 3, 3), padding=(1, 1))


    def _init_weights(self, m) -> None:
        # Gaussian initialization
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    # === SAFE DEBUG UTILS ===

    @torch._dynamo.disable # 이게 있어야 모델 자체에 로깅 기능을 넣어도 `torch.compile`에서 오류가 덜 발생하게 되는 듯.
    def open_debug_log(self) -> io.TextIOWrapper:
        os.makedirs('/workspace/logs', exist_ok=True)
        return open('/workspace/logs/ReBotNetLog.txt', 'a+')

    @torch._dynamo.disable
    def safe_log_tensor_stats(self, tensor: Tensor, name: str, logf) -> None:
        print(f"[DEBUG] After {name}: Max {tensor.max()}, Min {tensor.min()}", file=logf, flush=True)

    @torch._dynamo.disable
    def safe_save_tensor_image(self, tensor: Tensor, name: str) -> None:
        os.makedirs('/workspace/figures', exist_ok=True)
        patch = tensor[0].clone().detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.title(f'Model {name}')
        plt.imshow(torch.clamp(patch.to(dtype=torch.float32).permute(1, 2, 0), min=0, max=1).numpy())
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f'/workspace/figures/ReBotNet_{name}.png')
        plt.close()

    # === MODEL FORWARD METHODS ===

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, D, H, W)
        x.transpose_(1, 2).requires_grad_(True)
        logf = None
        if self.debug:
            logf = self.open_debug_log()
            print("\n[DEBUG] **Input Tensor Stats**", file=logf, flush=True)
            print(f"  - Max input: {x.max()}, Min input: {x.min()}", file=logf, flush=True)
            print(f"  - NaN in input: {x.isnan().any()}, Inf in input: {x.isinf().any()}", file=logf, flush=True)

        # x_org is pure input tensor, and x[:, 0, ...] is assumed as a previous prediction.
        x_org = x[:, 1, ...].clone()
        # Use both for computation.
        x = rearrange(x, 'b t c h w -> b (t c) h w').contiguous()

        # Embedding
        x_1 = self.big_embedding1(x[:, 0:3, ...])
        x_2 = self.big_embedding2(x[:, 3:6, ...])
        x_temp = torch.cat((x_1, x_2), dim=1)

        if self.debug:
            print(f"[DEBUG] After Embeddings: x_1 {x_1.shape}, x_2 {x_2.shape}, concat {x_temp.shape}", file=logf, flush=True)

        # Downsampling by strided convolutions
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
                if self.debug:
                    print(f"[DEBUG] Stage {i}: Output Shape {x_out.shape}", file=logf, flush=True)

        ############### Bottleneck parts ###############
        x = x_out # Downsampled embeddings
        x_size = x.size() # For the Shape inference
        x_temp = self.pool(x_temp.transpose(1, 2).contiguous()) # Maxpooling
        x_temp = self.temporal_transformer(x_temp) # Not Transoformer, but a Bottleneck Mixer.

        if self.debug:
            print(f"[DEBUG] Bottleneck Input Sizes - x: {x.shape}, x_temp: {x_temp.shape}", file=logf, flush=True)

        x = self.to_patch_embedding(x) # Embed along different dimensions
        x = self.bottleneck(x.transpose(1, 2).contiguous()) # Also a Bottleneck Mixer
        x = x + x_temp # Residual connection

        if self.debug:
            print(f"[DEBUG] Bottleneck Output Shape: {x.shape}", file=logf, flush=True)

        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3]) # Reconstruct the orignal shape

        # Upsampling
        x = self.upsample2(x)
        x = self.chchange1(x) + outs[-2]
        x = self.upsample3(x)
        x = self.chchange2(x) + outs[-3]
        x = self.upsample4(x)
        x = self.chchange3(x) + outs[-4]

        # Final Upscaling
        if self.upscale == 1:
            x = self.upsamplef1(x)
            x = self.conv_last(x)
            output = x + x_org
        else:
            x = self.upsamplef1(x)
            x = self.upsamplef2(x)
            x = self.conv_last(x)
            output = x + F.interpolate(
                x_org, size=(128, 128), mode='bilinear', align_corners=False
            )

        if self.debug:
            print(f"[DEBUG] Final Output Shape: {output.shape}", file=logf, flush=True)

            # Debug visualization
            stages: dict[str, Tensor] = {
                "input": x_org,
                "final_output": output
            }
            for name, tensor in stages.items():
                self.safe_log_tensor_stats(tensor, name=name, logf=logf)
                self.safe_save_tensor_image(tensor, name=name)

            logf.close()

        return output


if __name__ == '__main__':
    from torchinfo import summary
    model = ReBotNet()
    summary(model, input_size=(1, 3, 2, 384, 384), device='cpu')