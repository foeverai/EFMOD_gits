from idlelib.configdialog import changes

import torch
import torch.nn as nn
import math
from functools import partial
from torch.nn import functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from typing import Optional

from mmseg.models.backbones.resnet import ResNetV1d
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm


'''===================================DConv=========================================='''
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
class ACBlock(BaseModule):
    def __init__(self,
                 in_planes:int,
                 out_planes:int,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(ACBlock, self).__init__(init_cfg)
        self.squre = ConvModule(in_planes, out_planes, kernel_size=3, padding=1, stride=1,norm_cfg=None,act_cfg=None)
        self.cross_ver = ConvModule(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1,norm_cfg=None,act_cfg=None)
        self.cross_hor = ConvModule(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1,norm_cfg=None,act_cfg=None)
        _,self.bn = build_norm_layer(norm_cfg,out_planes)
        self.ReLU = build_activation_layer(act_cfg)

    def forward(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3))

'''===================================CA=============================================='''
class h_sigmoid(BaseModule):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(BaseModule):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(BaseModule):
    def __init__(self,
                 inp,
                 oup,
                 reduction=32,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(CoordAtt, self).__init__(init_cfg)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = ConvModule(inp, mip, kernel_size=1, stride=1, padding=0,norm_cfg=norm_cfg,act_cfg=None)
        self.act = h_swish()

        self.conv_h = ConvModule(mip, oup, kernel_size=1, stride=1, padding=0,norm_cfg=None,act_cfg=None)
        self.conv_w = ConvModule(mip, oup, kernel_size=1, stride=1, padding=0,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
'''========================================================================================='''
class MCB(BaseModule):
    def __init__(self,
                 in_chan,
                 out_chan,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None):
        super(MCB, self).__init__(init_cfg)
        k_size = [3,5,7,9]
        self.mc1 = ConvModule(in_chan,out_chan,kernel_size=k_size[0],stride=1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc2 = ConvModule(in_chan,out_chan,kernel_size=k_size[1],stride=1,padding=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc3 = ConvModule(in_chan,out_chan,kernel_size=k_size[2],stride=1,padding=3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc4 = ConvModule(in_chan,out_chan,kernel_size=k_size[3],stride=1,padding=4,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.incept = ConvModule(in_chan,out_chan,kernel_size=1,stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.con = ConvModule(out_chan*5,out_chan,kernel_size=1,stride=1,padding=0,norm_cfg=None,act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.mc1(x)
        x2 = self.mc2(x)
        x3 = self.mc3(x)
        x4 = self.mc4(x)
        x5 = self.incept(x)
        out = self.con(torch.cat([x1, x2, x3, x4, x5], dim=1))
        return out





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
'''========================================================================================='''
class MCB(BaseModule):
    def __init__(self,
                 in_chan,
                 out_chan,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None):
        super(MCB, self).__init__(init_cfg)
        k_size = [3,5,7,9]
        self.mc1 = ConvModule(in_chan,out_chan,kernel_size=k_size[0],stride=1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc2 = ConvModule(in_chan,out_chan,kernel_size=k_size[1],stride=1,padding=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc3 = ConvModule(in_chan,out_chan,kernel_size=k_size[2],stride=1,padding=3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc4 = ConvModule(in_chan,out_chan,kernel_size=k_size[3],stride=1,padding=4,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.incept = ConvModule(in_chan,out_chan,kernel_size=1,stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.con = ConvModule(out_chan*5,out_chan,kernel_size=1,stride=1,padding=0,norm_cfg=None,act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.mc1(x)
        x2 = self.mc2(x)
        x3 = self.mc3(x)
        x4 = self.mc4(x)
        x5 = self.incept(x)
        out = self.con(torch.cat([x1, x2, x3, x4, x5], dim=1))
        return out


'''======================================================================================================================'''
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
            # self.conv = nn.Sequential(
            #     ACBlock(in_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg),
            #     ACBlock(out_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg),
            # )
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


# # @MODELS.register_module()
# class FCR_Abl_all(BaseModule):
#     def __init__(self,
#                  in_channels:int=3,
#                  base_channels:int=32,
#                  norm_eval=False,
#                  norm_cfg: Optional[dict] = dict(type='BN'),
#                  act_cfg: Optional[dict] = dict(type='ReLU'),
#                  init_cfg: Optional[dict] = None
#                  ):
#         super(FCR_Abl_all, self).__init__(init_cfg)
#         self.band_num = in_channels
#         self.class_num = base_channels
#         self.norm_eval = norm_eval
#
#         # channels = [32, 64, 128, 256, 512]
#         channels = [32, 64, 128, 256, 512]
#         ''' torch.Size([1, 64, 256, 256])
#             torch.Size([1, 64, 128, 128])
#             torch.Size([1, 128, 64, 64])
#             torch.Size([1, 256, 32, 32])
#             torch.Size([1, 512, 16, 16])'''
#         self.inc = DoubleConv(in_channels, channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)  # 第一层只有DoubleConv
#         # b,32,512,512
#
#         self.conv1 = Down(channels[0],channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)     # 64,256,256
#         self.conv2 = Down(channels[1],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)     # 128,128,128
#         self.conv3 = Down(channels[2],channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)     # 256,64,64
#         self.conv4 = Down(channels[3],channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)     # 512,32,32
#
#
#         self.deconv4 = Up(channels[4]+channels[3],channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)      #512+256->256,s32-64
#         self.deconv3 = Up(channels[3]+channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)    # 256+128->128,s64-128
#         self.deconv2 = Up(channels[2]+channels[1],channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)    # 128+64->64,s128-256
#         self.deconv1 = Up(channels[1]+channels[0],channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)   # 64+32->64,s256-512
#         self.deconv0 = nn.Sequential(
#                         DoubleConv(channels[0], base_channels, norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg)
#                         )
#
#     def forward(self, x):
#         out = []
#         e0 = self.inc(x)                # b,32,512,512
#         e1 = self.conv1(e0)             # 64,256,256
#         e2 = self.conv2(e1)             # 128,128,128
#         e3 = self.conv3(e2)             # 256,64,64
#         e4 = self.conv4(e3)             # 512,32,32
#         out.append(e4)                  # 0
#
#         d1 = self.deconv4(e4,e3)        # 256,64,64
#         out.append(d1)                  # 1
#
#         d2 = self.deconv3(d1,e2)        # 128,128,128
#         out.append(d2)                  # 2
#
#         d3 = self.deconv2(d2,e1)        # 64,256,256
#         out.append(d3)                  # 3
#         # print(d3.shape)
#
#         d4 = self.deconv1(d3,e0)           # 32,512,512
#         d5 = self.deconv0(d4)           # 32,512,512
#         # print(d5.shape)
#         out.append(d5)                  # 4
#
#         return out
#
#     def train(self, mode=True):
#         """Convert the model into training mode while keep normalization layer
#         freezed."""
#         super().train(mode)
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 # trick: eval have effect on BatchNorm only
#                 if isinstance(m, _BatchNorm):
#                     m.eval()

# @MODELS.register_module()
class FCR_abl_fcr(BaseModule):
    def __init__(self,
                 in_channels:int=3,
                 base_channels:int=32,
                 norm_eval=False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(FCR_abl_fcr, self).__init__(init_cfg)
        self.band_num = in_channels
        self.class_num = base_channels
        self.norm_eval = norm_eval

        # channels = [32, 64, 128, 256, 512]
        channels = [32, 64, 128, 256, 512]
        # self.encoder = ResNetV1d(depth=34)
        self.inc = nn.Sequential(
            ACBlock(in_channels, channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),  # 第一层只有DoubleConv
            ACBlock(channels[0],channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        )       # b,32,512,512

        #=====================================
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[0], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[1], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 64,256,256

        #=====================================
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 128,128,128

        #====================================
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[3], channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[3], channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 256,64,64

        #=====================================
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[3], channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg),
            ACBlock(channels[4], channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg),
            ACBlock(channels[4], channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg),
        )# 512,32,32

        self.deconv4 = Up(channels[4]+channels[3],channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)      # 512+256->256,s32-64
        self.deconv3 = Up(channels[3]+channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)      # 256+128->128,s64-128
        self.deconv2 = Up(channels[2]+channels[1],channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)      # 128+64->64,s128-256
        self.deconv1 = Up(channels[1]+channels[0],channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)      # 64+32->32,s256-512
        self.deconv0 = nn.Sequential(
                        ACBlock(channels[0],base_channels,norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),
                        DoubleConv(base_channels, base_channels, norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg)
                        )

    def forward(self, x):
        out = []
        e0 = self.inc(x)        # 32,512,512
        e1 = self.conv1(e0)     # 64,256,256
        e2 = self.conv2(e1)     # 128,128,128
        e3 = self.conv3(e2)     # 256,64,64
        e4 = self.conv4(e3)     # 512,32,32
        out.append(e4)          # 0

        d1 = self.deconv4(e4,e3)    # 256,64,64
        out.append(d1)              # 1

        d2 = self.deconv3(d1,e2)    # 128,128,128
        out.append(d2)              # 2

        d3 = self.deconv2(d2,e1)    # 64,256,256
        out.append(d3)              # 3
        d4 = self.deconv1(d3,e0)    # 32,512,512
        d5 = self.deconv0(d4)       # 32,512,512
        out.append(d5)              # 4
        # print(d5.shape)

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
    from ptflops import get_model_complexity_info
    x = torch.randn(1,3,512,512).cuda()
    net = FCR_abl_fcr(in_channels=3,base_channels=32).cuda()
    out = net(x)
    # net = ResNetV1d(depth=34).cuda()

    # summary(net, input_size=(4, 3, 1500, 1500))
    macs, params = get_model_complexity_info(net, (3, 512, 512), print_per_layer_stat=False)
    print(macs, params)

