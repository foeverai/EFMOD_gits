#  python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:24:35 2020

@author: gis
https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge/blob/master/networks/dinknet.py

[1] L. Zhou, C. Zhang, and M. Wu, “D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, USA: IEEE, Jun. 2018, pp. 192–1924. doi: 10.1109/CVPRW.2018.00034.

"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
from mmseg.registry import MODELS
from pytorch_wavelets import DTCWTForward,DTCWTInverse
from mmcv.cnn.bricks import DropPath
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmseg.models.utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from functools import partial
from torchvision import models
# from collections import OrderedDict

# __all__ = ['BuildLinkNet34']
'''
DinkNet34_less_pool
DinkNet34
DinkNet50
DinkNet101
'''
nonlinearity = partial(F.relu, inplace=True)


# print(torch.__version__)

class Dblock_more_dilate(BaseModule):
    def __init__(self,
                 channel:int,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(Dblock_more_dilate, self).__init__(init_cfg)
        self.dilate1 = ConvModule(channel, channel, kernel_size=3, dilation=1, padding=1,norm_cfg=None,act_cfg=None)
        self.dilate2 = ConvModule(channel, channel, kernel_size=3, dilation=2, padding=2,norm_cfg=None,act_cfg=None)
        self.dilate3 = ConvModule(channel, channel, kernel_size=3, dilation=4, padding=4,norm_cfg=None,act_cfg=None)
        self.dilate4 = ConvModule(channel, channel, kernel_size=3, dilation=8, padding=8,norm_cfg=None,act_cfg=None)
        self.dilate5 = ConvModule(channel, channel, kernel_size=3, dilation=16, padding=16,norm_cfg=None,act_cfg=None)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(BaseModule):
    def __init__(self,
                 channel:int,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(Dblock, self).__init__(init_cfg)
        self.dilate1 = ConvModule(channel, channel, kernel_size=3, dilation=1, padding=1,norm_cfg=None,act_cfg=None)
        self.dilate2 = ConvModule(channel, channel, kernel_size=3, dilation=2, padding=2,norm_cfg=None,act_cfg=None)
        self.dilate3 = ConvModule(channel, channel, kernel_size=3, dilation=4, padding=4,norm_cfg=None,act_cfg=None)
        self.dilate4 = ConvModule(channel, channel, kernel_size=3, dilation=8, padding=8,norm_cfg=None,act_cfg=None)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class Dblock_test(BaseModule):
    def __init__(self,
                 channel:int,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(Dblock_test, self).__init__(init_cfg)
        self.dilate1 = ConvModule(channel, channel, kernel_size=3, dilation=1, padding=1,norm_cfg=None,act_cfg=None)
        self.dilate2 = ConvModule(channel, channel, kernel_size=3, dilation=2, padding=2,norm_cfg=None,act_cfg=None)
        self.dilate3 = ConvModule(channel, channel, kernel_size=3, dilation=4, padding=4,norm_cfg=None,act_cfg=None)
        self.dilate4 = ConvModule(channel, channel, kernel_size=3, dilation=8, padding=8,norm_cfg=None,act_cfg=None)
        self.dilate5 = ConvModule(channel, channel, kernel_size=3, dilation=16, padding=16,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class DecoderBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(DecoderBlock, self).__init__(init_cfg)

        self.conv1 = ConvModule(in_channels, in_channels // 4, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        _,self.norm2 = build_norm_layer(norm_cfg,in_channels // 4)
        self.relu2 = build_activation_layer(act_cfg)

        self.conv3 = ConvModule(in_channels // 4, n_filters, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


# class DinkNet34_less_pool(BaseModule):
#     def __init__(self,
#                  num_classes=1,
#                  norm_cfg: Optional[dict] = dict(type='BN'),
#                  act_cfg: Optional[dict] = dict(type='ReLU'),
#                  init_cfg: Optional[dict] = None,
#                  ):
#         super(DinkNet34_less_pool, self).__init__(init_cfg)
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=False)  # pretrained 是否加载预训练模型
#
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#
#         self.dblock = Dblock_more_dilate(256,norm_cfg=norm_cfg,act_cfg=act_cfg)
#
#         self.decoder3 = DecoderBlock(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
#         self.decoder2 = DecoderBlock(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
#         self.decoder1 = DecoderBlock(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = ConvModule(32, num_classes, 3, padding=1,norm_cfg=None,act_cfg=None)
#         if num_classes == 1:
#             self.last_activation = torch.nn.Sigmoid()
#         else:
#             self.last_activation = torch.nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#
#         # Center
#         e3 = self.dblock(e3)
#
#         # Decoder
#         d3 = self.decoder3(e3) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#
#         # Final Classification
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return self.last_activation(out)


# class DinkNet18(nn.Module):
#     def __init__(self, num_classes=1, Pretrained=False):
#         super(DinkNet18, self).__init__()
#
#         filters = [64, 128, 256, 512]
#
#         resnet = models.resnet18(pretrained=Pretrained)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock(512)  # 该模块 默认加载
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#         if num_classes == 1:
#             self.last_activation = torch.nn.Sigmoid()
#         else:
#             self.last_activation = torch.nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Center
#         e4 = self.dblock(e4)
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return self.last_activation(out)

@MODELS.register_module()
class DlinkNet34(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=32,
                 pretrained=False,
                 norm_eval: bool = False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = [dict(type='Kaiming', layer='Conv2d',
                                                  a=math.sqrt(5),
                                                  distribution='uniform',
                                                  mode='fan_in',
                                                  nonlinearity='leaky_relu'),
                                             dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])]  # 初始化配置字典
                 ):
        super(DlinkNet34, self).__init__(init_cfg)
        self.norm_eval = norm_eval

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.decoder4 = DecoderBlock(filters[3], filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = DecoderBlock(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = DecoderBlock(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = DecoderBlock(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = ConvModule(32, base_channels, 3, padding=1,norm_cfg=None,act_cfg=None)

        # if num_classes == 1:
        #     self.last_activation = torch.nn.Sigmoid()
        # else:
        #     self.last_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = []
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        output.append(d4)               # index = 0
        d3 = self.decoder3(d4) + e2
        output.append(d3)               # index = 1
        d2 = self.decoder2(d3) + e1
        output.append(d2)               # index = 2
        d1 = self.decoder1(d2)
        output.append(d1)               # index = 3  [bs,64,w/2,h/2]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        output.append(out)              # index = 4 [bs,32,w,h]

        return output

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


# class DinkNet50(BaseModule):
#
#     def __init__(self,
#                  num_classes=1,
#                  norm_cfg: Optional[dict] = dict(type='BN'),
#                  act_cfg: Optional[dict] = dict(type='ReLU'),
#                  init_cfg: Optional[dict] = None,
#                  ):
#         super(DinkNet50, self).__init__(init_cfg)
#
#         filters = [256, 512, 1024, 2048]
#         resnet = models.resnet50(pretrained=False)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock_more_dilate(2048,norm_cfg=norm_cfg,act_cfg=act_cfg)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
#         self.decoder3 = DecoderBlock(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
#         self.decoder2 = DecoderBlock(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
#         self.decoder1 = DecoderBlock(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = ConvModule(32, num_classes, 3, padding=1,norm_cfg=None,act_cfg=None)
#
#         if num_classes == 1:
#             self.last_activation = torch.nn.Sigmoid()
#         else:
#             self.last_activation = torch.nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         print(x.shape)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Center
#         e4 = self.dblock(e4)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return self.last_activation(out)


# class DinkNet101(nn.Module):
#     def __init__(self, num_classes=1):
#         super(DinkNet101, self).__init__()
#
#         filters = [256, 512, 1024, 2048]
#         resnet = models.resnet101(pretrained=False)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock_more_dilate(2048)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#         if num_classes == 1:
#             self.last_activation = torch.nn.Sigmoid()
#         else:
#             self.last_activation = torch.nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Center
#         e4 = self.dblock(e4)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return self.last_activation(out)


if __name__ == '__main__':
    '''model_list:
        net = DinkNet34_less_pool()
        net = DinkNet34()
        net = DinkNet50()
        net = DinkNet101()
        net = LinkNet34() 
        net = ReNetLinkNet34()
        net = BuildLinkNet34()
        net = BuildLinkNet34()

        summary(net.cuda(), (3,512 ,512))

    '''
    x = torch.randn(1,3,64,64).cuda()
    net = DlinkNet34(base_channels=32).cuda()  # Trainable params: 11,548,737
    # net = LinkNet34(num_classes=1) # Trainable params: 21,656,897
    out_list = net(x)

