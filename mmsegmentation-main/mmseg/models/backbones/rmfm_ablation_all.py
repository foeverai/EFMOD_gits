# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from pytorch_wavelets import DTCWTForward,DWTForward
from mmseg.registry import MODELS

"""-----------------------------------------------------------------------------"""
class DoubleConv(BaseModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            ConvModule(in_channels, mid_channels,kernel_size=3, padding=1,bias=False,norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
        )

    def forward(self, x):
        return self.double_conv(x)

""""""

class Down(BaseModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)

""""""
class ASPPConv(nn.Sequential):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 dilation:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        modules = [
            ConvModule(in_channels, out_channels,3,padding=dilation,dilation=dilation,bias=False,norm_cfg=None,act_cfg=None),
            ConvModule(out_channels,out_channels,(9,1),1,padding=(4,0),norm_cfg=None,act_cfg=None),
            ConvModule(out_channels,out_channels,(1,9),1,(0,4),norm_cfg=norm_cfg,act_cfg=act_cfg),
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            ConvModule(in_channels,out_channels,1,bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构

class ASPP(BaseModule):
    def __init__(self,
                 in_channels:int,
                 atrous_rates:list,
                 out_channels:int=256,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None
                 ):
        super(ASPP, self).__init__(init_cfg)
        modules = []
        # 1*1 卷积
        modules.append(
            ConvModule(in_channels,out_channels,1,bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)
)

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate,norm_cfg=norm_cfg,act_cfg=act_cfg))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            ConvModule(len(self.convs) * out_channels, out_channels, 1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


''''''


class Up(BaseModule):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 bilinear:bool=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
''''''


class OutConv(BaseModule):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super(OutConv, self).__init__(init_cfg)
        self.conv = ConvModule(in_channels, out_channels, kernel_size=1,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        return self.conv(x)



# @MODELS.register_module()
class RMFMNet_ablation_all(BaseModule):
    def __init__(self,
                 in_channels:int=3,
                 base_channels:int=64,
                 bilinear=False,
                 norm_eval:bool=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super(RMFMNet_ablation_all, self).__init__(init_cfg)
        self.input_channel = in_channels
        self.class_num = base_channels
        self.norm_eval = norm_eval

        out_channels = [32, 64, 128, 256, 512]

        self.conv1 = DoubleConv(self.input_channel, out_channels[0], norm_cfg=norm_cfg,
                                act_cfg=act_cfg)  # (bs,c,h,w)→(bs,32,h,w)

        self.conv2 = nn.Sequential(
            # (bs,32,h,w)→(bs,32,h/2,w/2)
            Down(out_channels[0], out_channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )  # (bs,32,h,w)→(bs,64,h/2,w/2)
        self.conv3 = nn.Sequential(
            # (bs,64,h/2,w/2)→(bs,64,h/4,w/4)
            Down(out_channels[1], out_channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )  # (bs,64,h/4,w/4)→(bs,128,h/4,w/4)

        self.conv4 = nn.Sequential(
            # (bs,128,h/4,w/4)→(bs,128,h/8,w/8)
            Down(out_channels[2], out_channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )  # (bs,128,h/4,w/4)→(bs,256,h/8,w/8)

        self.conv5 = nn.Sequential(
            # (bs,256,h/8,w/8)→(bs,256,h/16,w/16)
            Down(out_channels[3], out_channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )  # (bs,256,h/8,w/8)→(bs,512,h/16,w/16).

        self.deconv5 = ASPP(out_channels[4], [1, 6, 12, 18], out_channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg)

        factor = 2 if bilinear else 1  # 默认bilinear==False
        self.up_1 = Up(512, 256 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_2 = Up(256, 128 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_3 = Up(128, 64 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_4 = Up(64, 32 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.final_conv = ConvModule(out_channels[0], self.class_num, kernel_size=1, stride=1, norm_cfg=None,
                                     act_cfg=None)  # (bs,32,h,w)→(bs,num_classes,h,w)

    def forward(self, x):
        out = []
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.deconv5(x5)
        out.append(x5)
        x = self.up_1(x5, x4)
        out.append(x)
        x = self.up_2(x, x3)
        out.append(x)
        x = self.up_3(x, x2)
        out.append(x)
        x = self.up_4(x, x1)
        logits = self.final_conv(x)
        out.append(logits)
        return out

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == '__main__':
    x = torch.randn(2,3,64,64).cuda()
    model = RMFMNet_ablation_all(3,32).cuda()
    print(model(x)[4].shape)

