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
class Down_db1(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None
                 ):
        super(Down_db1, self).__init__(init_cfg)
        self.wt = DWTForward(J=1, mode='zero', wave='db1')

        self.conv_bn_relu = ConvModule(in_ch*4,out_ch,1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x


class DTCWT_layer(BaseModule):
    def __init__(self,
                 in_ch:int,
                 out_ch:int,
                 j:int=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None
                 ):
        super(DTCWT_layer, self).__init__(init_cfg)
        self.j = j
        self.wt = DTCWTForward(J=j, biort='near_sym_a', qshift='qshift_a')
        # self.conv_ = ConvModule(in_ch,in_ch,kernel_size=3,stride=1,padding=1,groups=in_ch,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.conv1_1 = nn.Conv2d(in_ch, in_ch, kernel_size=1,stride=1)
        self.conv1_1 = ConvModule(in_ch,in_ch,1,1,norm_cfg=None,act_cfg=None)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.conv_bn_relu = ConvModule(in_ch,out_ch,1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)
    def forward(self, x):
        """
        yl
        ([1,3,16,16])
        yh
        torch.Size([1, 3, 6, 32, 32, 2])
        torch.Size([1, 3, 6, 16, 16, 2])
        torch.Size([1, 3, 6, 8, 8, 2])
        """
        yL, yH = self.wt(x)
        # print(yL.shape)
        abs_ele = []
        num = len(yH)
        # 取模
        for i in range(num):
            real_part = yH[i][...,0]
            imag_part = yH[i][...,1]
            complex_y = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))
            abs_ele.append(torch.abs(complex_y))
        y_15 = abs_ele[0][:,:,0,::]
        y_45 = abs_ele[0][:,:,1,::]
        y_75 = abs_ele[0][:,:,2,::]
        y_105 = abs_ele[0][:,:,3,::]
        y_135 = abs_ele[0][:,:,4,::]
        y_165 = abs_ele[0][:,:,5,::]
        temp_out = self.conv1_1(y_15+y_45+y_75+y_105+y_135+y_165)
        # print(out.shape)
        out = self.upsample(self.conv_bn_relu(temp_out+self.conv1_1(yL)))
        return out
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
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
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
            DTCWT_layer(in_channels, in_channels,2, norm_cfg=norm_cfg,act_cfg=act_cfg),
            Down_db1(in_channels, out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

    def forward(self, x):
        return self.down_conv(x)

""""""


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



@MODELS.register_module()
class UNetdct(BaseModule):
    def __init__(self,
                 in_channels:int=3,
                 base_channels:int=64,
                 bilinear=False,
                 norm_eval:bool=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super(UNetdct, self).__init__(init_cfg)
        self.norm_eval = norm_eval
        self.n_channels = in_channels
        self.n_classes = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 第一层只有DoubleConv

        self.down1 = Down(64, 128,norm_cfg=norm_cfg,act_cfg=act_cfg)  # max_pool + DoubleConv生成左侧第二层
        self.down2 = Down(128, 256,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 同上生成左侧第三层
        self.down3 = Down(256, 512,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 同上生成左侧第四层
        factor = 2 if bilinear else 1  # 默认bilinear==False
        self.down4 = Down(512, 1024 // factor,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 生成最下面一层

        self.up1 = Up(1024, 512 // factor, bilinear,norm_cfg=norm_cfg,act_cfg=act_cfg)  # TransposedConv + copy_crop + DoubleConv生成右侧分支第四层
        self.up2 = Up(512, 256 // factor, bilinear,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 同上，生成右侧第三层
        self.up3 = Up(256, 128 // factor, bilinear,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 同上，生成右侧第二层
        self.up4 = Up(128, 64, bilinear,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 同上，生成右侧一层

        self.outc = OutConv(64, base_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)  # 输出层

    def forward(self, x):
        out = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out.append(x5)
        x = self.up1(x5, x4)
        out.append(x)
        x = self.up2(x, x3)
        out.append(x)
        x = self.up3(x, x2)
        out.append(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
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
    x = torch.randn(1,3,64,64).cuda()
    model = UNetdct(3,64).cuda()
    print(model(x).shape)

