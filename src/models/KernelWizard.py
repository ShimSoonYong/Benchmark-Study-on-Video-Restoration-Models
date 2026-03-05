import torch
import functools
from torch import nn, Tensor
from typing import Any, Union
import torch.nn.init as init
import torch.nn.functional as F

""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels:int, out_channels:int, mid_channels:Union[bool, None]=None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.double_conv(x)


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the Unet submodule with skip connection.  

    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|

    """

    def __init__(
        self,
        outer_nc:int,
        inner_nc:int,
        input_nc:int=None,
        submodule:Union[nn.Module, None]=None,
        outermost:bool=False,
        innermost:bool=False,
        norm_layer:nn.Module=nn.BatchNorm2d,
        use_dropout:nn.Module=False,
    ) -> None:
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) --previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            up = [uprelu, upconv, nn.Tanh()]
            down = [downconv]
            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]

            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:

        if self.outermost:
            return self.up(self.submodule(self.down(x), noise))
        elif self.innermost:  # add skip connections
            if noise is None:
                noise = torch.randn((1, 512, 8, 8)).cuda() * 0.0007
            return torch.cat((self.up(torch.cat((self.down(x), noise), dim=1)), x), dim=1)
        else:
            return torch.cat((self.up(self.submodule(self.down(x), noise)), x), dim=1)

"""
Model util functions
"""

class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


def get_norm_layer(norm_type="instance") -> nn.Module:
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization
                            layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and
    track running statistics (mean/stddev).

    For InstanceNorm, we do not use learnable affine
    parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError(
            f"normalization layer {norm_type}\
                                    is not found"
        )
    return norm_layer


def initialize_weights(net_l:nn.Module, scale=1) -> None:
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

"""
ResNet Backbone
"""

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim: int, padding_type :str, norm_layer: nn.Module, use_dropout:bool, use_bias:bool) -> None:
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim: int, padding_type :str, norm_layer: nn.Module, use_dropout:bool, use_bias:bool) -> nn.Sequential:
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding
                                   layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
                              and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                        is not implemented"
            )

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                      is not implemented"
            )
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x:Tensor) -> Tensor:
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf:int=64) -> None:
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x:Tensor) -> Tensor:
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out

"""
Main model
"""


# The function F in the paper
class KernelExtractor(nn.Module):
    def __init__(self, opt:dict) -> None:
        super(KernelExtractor, self).__init__()

        nf = opt["nf"]
        self.kernel_dim = opt["kernel_dim"]
        self.use_sharp = opt["KernelExtractor"]["use_sharp"]
        self.use_vae = opt["use_vae"]

        # Blur estimator
        norm_layer = get_norm_layer(opt["KernelExtractor"]["norm"])
        n_blocks = opt["KernelExtractor"]["n_blocks"]
        padding_type = opt["KernelExtractor"]["padding_type"]
        use_dropout = opt["KernelExtractor"]["use_dropout"]
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        input_nc = nf * 2 if self.use_sharp else nf
        output_nc = self.kernel_dim * 2 if self.use_vae else self.kernel_dim

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(nf),
            nn.ReLU(True),
        ]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            inc = min(nf * mult, output_nc)
            ouc = min(nf * mult * 2, output_nc)
            model += [
                nn.Conv2d(inc, ouc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(nf * mult * 2),
                nn.ReLU(True),
            ]

        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    output_nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        self.model = nn.Sequential(*model)

    def forward(self, sharp:Tensor, blur:Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(torch.cat((sharp, blur), dim=1))
        if self.use_vae:
            return output[:, : self.kernel_dim, :, :], output[:, self.kernel_dim :, :, :]

        return output, torch.zeros_like(output)#.cuda()


# The function G in the paper
class KernelAdapter(nn.Module):
    def __init__(self, opt:dict) -> None:
        super(KernelAdapter, self).__init__()
        input_nc = opt["nf"]
        output_nc = opt["nf"]
        ngf = opt["nf"]
        norm_layer = get_norm_layer(opt["Adapter"]["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x:Tensor, k:Tensor) -> Tensor:
        """Standard forward"""
        return self.model(x, k)


class KernelWizard(nn.Module):
    def __init__(self, opt: dict) -> None:
        super(KernelWizard, self).__init__()
        lrelu = nn.LeakyReLU(negative_slope=0.1)
        front_RBs = opt["front_RBs"]
        back_RBs = opt["back_RBs"]
        num_image_channels = opt["input_nc"]
        nf = opt["nf"]

        # Features extraction
        resBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        feature_extractor = []

        feature_extractor.append(nn.Conv2d(num_image_channels, nf, 3, 1, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)

        for i in range(front_RBs):
            feature_extractor.append(resBlock_noBN_f())

        self.feature_extractor = nn.Sequential(*feature_extractor)

        # Kernel extractor
        self.kernel_extractor = KernelExtractor(opt)

        # kernel adapter
        self.adapter = KernelAdapter(opt)

        # Reconstruction
        recon_trunk = []
        for i in range(back_RBs):
            recon_trunk.append(resBlock_noBN_f())

        # upsampling
        recon_trunk.append(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, num_image_channels, 3, 1, 1, bias=True))

        self.recon_trunk = nn.Sequential(*recon_trunk)

    def adaptKernel(self, x_sharp:Tensor, kernel:Tensor) -> Tensor:
        B, C, H, W = x_sharp.shape
        base = x_sharp

        x_sharp = self.feature_extractor(x_sharp)

        out = self.adapter(x_sharp, kernel)
        out = self.recon_trunk(out)
        out += base

        return out

    def extractKernel(self, x_sharp:Tensor, x_blur:Tensor) -> Tensor:
        x_sharp = self.feature_extractor(x_sharp)
        x_blur = self.feature_extractor(x_blur)

        output = self.kernel_extractor(x_sharp, x_blur)
        return output
    
    def forward(self, x_sharp: Tensor, x_blur: Tensor) -> Tensor:
        """Standard forward"""
        kernel: Tensor = self.extractKernel(x_sharp, x_blur)
        out: Tensor = self.adaptKernel(x_sharp, kernel)
        return out

if __name__ == "__main__":
    import os
    import yaml
    from torchinfo import summary

    with open(os.path.join("/workspace/options/KernelWizard_opts" + ".yaml"), "r") as f:
        opt: dict = yaml.load(f, Loader=yaml.FullLoader)
        print("option file loaded, opt_path: {}".format(os.path.join("options", "/workspace/options/KernelWizard_opts" + ".yaml")))
    opt = opt["model"]
    model = KernelWizard(opt)
    print(summary(model, input_size=((1, 3, 256, 256), (1, 3, 256, 256))))