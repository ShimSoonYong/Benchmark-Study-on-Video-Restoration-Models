import torch
from torch import nn, Tensor
import torch.nn.functional as F

"""
MIMO-UNet - Official Pytorch Implementation
https://github.com/chosj95/MIMO-UNet.git
"""


class BasicConv(nn.Module):
    def __init__(
        self, in_channel:int, out_channel:int, kernel_size:int, stride:int, 
        bias:bool=True, norm:bool=False, relu:bool=True, transpose:bool=False
    ) -> None:
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
            )
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel:int, num_res:int=8) -> None:
        super(EBlock, self).__init__()

        layers: list[ResBlock] = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x:Tensor)->Tensor:
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel:int, num_res:int=8) -> None:
        super(DBlock, self).__init__()

        layers: list[ResBlock] = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x1:Tensor, x2:Tensor, x4:Tensor) -> Tensor:
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane:int) -> None:
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True),
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel:int) -> None:
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

###############################################################################################################
###############################################################################################################
########################################### Main Modules ######################################################
###############################################################################################################
###############################################################################################################

class MIMOUNet(nn.Module):
    def __init__(self, debug:bool, num_res:int=8) -> None:
        super(MIMOUNet, self).__init__()
        self.debug: bool = debug
        if debug:
            self.feature_maps: dict = {}

        base_channel = 32

        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )

        self.feat_extract = nn.ModuleList(
            [
                BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
                BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.Decoder = nn.ModuleList(
            [DBlock(base_channel * 4, num_res), DBlock(base_channel * 2, num_res), DBlock(base_channel, num_res)]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
                BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1), AFF(base_channel * 7, base_channel * 2)])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x: Tensor) -> list[Tensor]:

        self.feature_maps.clear()  # Clear previous maps
        self.feature_maps["input"] = x

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []

        x_ = self.feat_extract[0](x)
        self.feature_maps["feat0"] = x_

        res1 = self.Encoder[0](x_)
        self.feature_maps["res1"] = res1

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        self.feature_maps["fam2"] = z

        res2 = self.Encoder[1](z)
        self.feature_maps["res2"] = res2

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        self.feature_maps["fam1"] = z

        z = self.Encoder[2](z)
        self.feature_maps["res3"] = z

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        self.feature_maps["res2_fused"] = res2

        res1 = self.AFFs[0](res1, z21, z41)
        self.feature_maps["res1_fused"] = res1

        z = self.Decoder[0](z)
        self.feature_maps["dec0"] = z

        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        self.feature_maps["dec1"] = z

        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        self.feature_maps["dec2"] = z

        z = self.feat_extract[5](z)
        outputs.append(z + x)

        self.feature_maps["output"] = z + x

        if self.debug:
            self.log_features(prefix="MIMOUNet")

        return outputs

    def log_features(self, prefix: str = "feat") -> None:
        import os
        import matplotlib.pyplot as plt

        os.makedirs("/workspace/figures", exist_ok=True)
        os.makedirs("/workspace/logs", exist_ok=True)

        with open("/workspace/logs/MIMOUNetLog.txt", "a+") as log_file:
            for name, feat in self.feature_maps.items():
                feat: Tensor = feat.detach().cpu()[0]  # (C, H, W)

                log_file.write(f"{name}:\n")
                log_file.write(f"  Shape: {feat.shape}\n")

                if name not in ["input", "output"]:
                    feat: Tensor = feat.mean(0) # Average over channels
                else:
                    feat: Tensor = feat.permute(1, 2, 0)
                
                stats: str = f"  Stats of {prefix}_{name}: min={feat.min():.6f}, max={feat.max():.6f}, mean={feat.mean():.6f}"
                log_file.write(stats + "\n")
                path: str = os.path.join("/workspace/figures", f"{prefix}_{name}.png")

                feat_norm: Tensor = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)  # Normalize for plotting
                plt.figure(figsize=(10, 10))
                plt.imshow(feat_norm.numpy(), cmap="viridis")
                plt.axis("off")

                if name not in ["input", "output"]:
                    plt.colorbar()

                plt.title(f"{prefix}_{name}")
                plt.savefig(path, bbox_inches="tight")
                plt.close()

                log_file.write("\n")



class MIMOUNetPlus(nn.Module):
    def __init__(self, debug: bool = False, num_res: int = 20) -> None:
        super(MIMOUNetPlus, self).__init__()
        self.debug: bool = debug
        if debug:
            self.feature_maps: dict = {}

        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2),
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.debug:
            self.feature_maps.clear()
            self.feature_maps["input"] = x

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        if self.debug: self.feature_maps["res1"] = res1

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        if self.debug: self.feature_maps["res2"] = res2

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        if self.debug: self.feature_maps["res3"] = z

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)
        if self.debug:
            self.feature_maps["res2_fused"] = res2
            self.feature_maps["res1_fused"] = res1

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        if self.debug: self.feature_maps["dec0"] = z

        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        if self.debug: self.feature_maps["dec1"] = z

        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        if self.debug: self.feature_maps["dec2"] = z

        z = self.feat_extract[5](z)
        outputs.append(z + x)

        if self.debug:
            self.feature_maps["output"] = z + x
            self.log_features(prefix="MIMOUNetPlus")

        return outputs

    def log_features(self, prefix: str = "feat") -> None:
        import os
        import matplotlib.pyplot as plt

        os.makedirs("/workspace/figures", exist_ok=True)
        os.makedirs("/workspace/logs", exist_ok=True)

        with open("/workspace/logs/MIMOUNetPlusLog.txt", "a+") as log_file:
            for name, feat in self.feature_maps.items():
                feat: Tensor = feat.detach().cpu()[0]
                log_file.write(f"{name}:\n")
                log_file.write(f"  Shape: {feat.shape}\n")

                if name not in ["input", "output"]:
                    feat: Tensor = feat.mean(0)
                else:
                    feat: Tensor = feat.permute(1, 2, 0)

                stats: str = f"Stats of {prefix}_{name}: min={feat.min():.6f}, max={feat.max():.6f}, mean={feat.mean():.6f}"
                log_file.write(stats + "\n")
                path: str = os.path.join("/workspace/figures", f"{prefix}_{name}.png")

                feat_norm: Tensor = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
                plt.figure(figsize=(10, 10))
                plt.imshow(feat_norm.numpy(), cmap="viridis")
                plt.axis("off")
                
                if name not in ["input", "output"]:
                    plt.colorbar()

                plt.title(f"{prefix}_{name}")
                plt.savefig(path, bbox_inches="tight")
                plt.close()

                log_file.write("\n")



if __name__ == "__main__":
    from torchinfo import summary
    model = MIMOUNet(debug=True)
    # model = MIMOUNetPlus()
    print(summary(model, (1, 3, 256, 256)))