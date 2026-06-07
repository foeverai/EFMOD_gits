import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

import os

# from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
# from aspp import ASPP, ASPP_Bottleneck
'''--------------------------------------------------------------------------------------------------------------------'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return ConvModule(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation,
                      norm_cfg=None,act_cfg=None)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return ConvModule(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,norm_cfg=None,act_cfg=None)


class Bottleneck(BaseModule):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.SyncBatchNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseModule):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        aux_x = self.layer3(x)
        x = self.layer4(aux_x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return aux_x,x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)



'''--------------------------------------------------backon-resnet----------------------------------------------------'''
# def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
#     strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])
#
#     blocks = []
#     for stride in strides:
#         blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
#         in_channels = block.expansion*channels
#
#     layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)
#
#     return layer
#
# class BasicBlock(BaseModule):
#     expansion = 1
#
#     def __init__(self,
#                  in_channels,
#                  channels,
#                  stride=1,
#                  dilation=1,
#                  norm_cfg: Optional[dict] = dict(type='BN'),
#                  act_cfg=dict(type='ReLU'),
#                  init_cfg=None
#                  ):
#         super(BasicBlock, self).__init__(init_cfg)
#
#         out_channels = self.expansion*channels
#
#         self.conv1 = ConvModule(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
#                                 bias=False,norm_cfg=None,act_cfg=None)
#         _,self.bn1 = build_norm_layer(norm_cfg,channels)
#
#         self.conv2 = ConvModule(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
#                                 bias=False,norm_cfg=None,act_cfg=None)
#         self.bn2 = build_norm_layer(norm_cfg,channels)
#
#         if (stride != 1) or (in_channels != out_channels):
#             conv = ConvModule(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,norm_cfg=None,act_cfg=None)
#             _,bn = build_norm_layer(norm_cfg,out_channels)
#             self.downsample = nn.Sequential(conv, bn)
#         else:
#             self.downsample = nn.Sequential()
#
#     def forward(self, x):
#         # (x has shape: (batch_size, in_channels, h, w))
#
#         out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
#         out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
#
#         out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
#
#         out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
#
#         return out
#
# class Bottleneck(BaseModule):
#     expansion = 4
#
#     def __init__(self,
#                  in_channels,
#                  channels,
#                  stride=1,
#                  dilation=1,
#                  norm_cfg: Optional[dict] = dict(type='BN'),
#                  act_cfg=dict(type='ReLU'),
#                  init_cfg=None
#                  ):
#         super(Bottleneck, self).__init__(init_cfg)
#
#         out_channels = self.expansion*channels
#
#         self.conv1 = ConvModule(in_channels, channels, kernel_size=1, bias=False,norm_cfg=None,act_cfg=None)
#         _,self.bn1 = build_norm_layer(norm_cfg,channels)
#
#         self.conv2 = ConvModule(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
#                                 bias=False, norm_cfg=None,act_cfg=None)
#         _,self.bn2 = build_norm_layer(norm_cfg,channels)
#
#         self.conv3 = ConvModule(channels, out_channels, kernel_size=1, bias=False,norm_cfg=None,act_cfg=None)
#         _,self.bn3 = build_norm_layer(norm_cfg,out_channels)
#
#         if (stride != 1) or (in_channels != out_channels):
#             conv = ConvModule(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,norm_cfg=None,act_cfg=None)
#             _,bn = build_norm_layer(norm_cfg,out_channels)
#             self.downsample = nn.Sequential(conv, bn)
#         else:
#             self.downsample = nn.Sequential()
#
#     def forward(self, x):
#         # (x has shape: (batch_size, in_channels, h, w))
#
#         out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w))
#         out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
#         out = self.bn3(self.conv3(out)) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)
#
#         out = out + self.downsample(x) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)
#
#         out = F.relu(out) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)
#
#         return out
#
# class ResNet_Bottleneck_OS16(BaseModule):
#     def __init__(self,
#                  num_layers,
#                  init_cfg=None
#                  ):
#         super(ResNet_Bottleneck_OS16, self).__init__(init_cfg)
#
#         if num_layers == 50:
#             resnet = models.resnet50()
#             # load pretrained model:
#             # resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet50-19c8e357.pth"))
#             # remove fully connected layer, avg pool and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-3])
#
#             print ("pretrained resnet, 50")
#         elif num_layers == 101:
#             resnet = models.resnet101()
#             # load pretrained model:
#             resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet101-5d3b4d8f.pth"))
#             # remove fully connected layer, avg pool and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-3])
#
#             print ("pretrained resnet, 101")
#         elif num_layers == 152:
#             resnet = models.resnet152()
#             # load pretrained model:
#             resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet152-b121ed2d.pth"))
#             # remove fully connected layer, avg pool and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-3])
#
#             print ("pretrained resnet, 152")
#         else:
#             raise Exception("num_layers must be in {50, 101, 152}!")
#
#         self.layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)
#
#     def forward(self, x):
#         # (x has shape (batch_size, 3, h, w))
#
#         # pass x through (parts of) the pretrained ResNet:
#         c4 = self.resnet(x) # (shape: (batch_size, 4*256, h/16, w/16)) (it's called c4 since 16 == 2^4)
#
#         output = self.layer5(c4) # (shape: (batch_size, 4*512, h/16, w/16))
#
#         return output
#
# class ResNet_BasicBlock_OS16(BaseModule):
#     def __init__(self, num_layers):
#         super(ResNet_BasicBlock_OS16, self).__init__()
#
#         if num_layers == 18:
#             resnet = models.resnet18()
#             # load pretrained model:
#             resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet18-5c106cde.pth"))
#             # remove fully connected layer, avg pool and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-3])
#
#             num_blocks = 2
#             print ("pretrained resnet, 18")
#         elif num_layers == 34:
#             resnet = models.resnet34()
#             # load pretrained model:
#             resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet34-333f7ec4.pth"))
#             # remove fully connected layer, avg pool and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-3])
#
#             num_blocks = 3
#             print ("pretrained resnet, 34")
#         else:
#             raise Exception("num_layers must be in {18, 34}!")
#
#         self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)
#
#     def forward(self, x):
#         # (x has shape (batch_size, 3, h, w))
#
#         # pass x through (parts of) the pretrained ResNet:
#         c4 = self.resnet(x) # (shape: (batch_size, 256, h/16, w/16)) (it's called c4 since 16 == 2^4)
#
#         output = self.layer5(c4) # (shape: (batch_size, 512, h/16, w/16))
#
#         return output
#
# class ResNet_BasicBlock_OS8(BaseModule):
#     def __init__(self, num_layers):
#         super(ResNet_BasicBlock_OS8, self).__init__()
#
#         if num_layers == 18:
#             resnet = models.resnet18()
#             # load pretrained model:
#             resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet18-5c106cde.pth"))
#             # remove fully connected layer, avg pool, layer4 and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-4])
#
#             num_blocks_layer_4 = 2
#             num_blocks_layer_5 = 2
#             print ("pretrained resnet, 18")
#         elif num_layers == 34:
#             resnet = models.resnet34()
#             # load pretrained model:
#             resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet34-333f7ec4.pth"))
#             # remove fully connected layer, avg pool, layer4 and layer5:
#             self.resnet = nn.Sequential(*list(resnet.children())[:-4])
#
#             num_blocks_layer_4 = 6
#             num_blocks_layer_5 = 3
#             print ("pretrained resnet, 34")
#         else:
#             raise Exception("num_layers must be in {18, 34}!")
#
#         self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)
#
#         self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)
#
#     def forward(self, x):
#         # (x has shape (batch_size, 3, h, w))
#
#         # pass x through (parts of) the pretrained ResNet:
#         c3 = self.resnet(x) # (shape: (batch_size, 128, h/8, w/8)) (it's called c3 since 8 == 2^3)
#
#         output = self.layer4(c3) # (shape: (batch_size, 256, h/8, w/8))
#         output = self.layer5(output) # (shape: (batch_size, 512, h/8, w/8))
#
#         return output
#
# def ResNet18_OS16():
#     return ResNet_BasicBlock_OS16(num_layers=18)
#
# def ResNet34_OS16():
#     return ResNet_BasicBlock_OS16(num_layers=34)
#
# def ResNet50_OS16():
#     return ResNet_Bottleneck_OS16(num_layers=50)
#
# def ResNet101_OS16():
#     return ResNet_Bottleneck_OS16(num_layers=101)
#
# def ResNet152_OS16():
#     return ResNet_Bottleneck_OS16(num_layers=152)
#
# def ResNet18_OS8():
#     return ResNet_BasicBlock_OS8(num_layers=18)
#
# def ResNet34_OS8():
#     return ResNet_BasicBlock_OS8(num_layers=34)
#
#
# '''--------------------------------------------ASPP--------------------------------------------------------------------'''
# class ASPP(BaseModule):
#     def __init__(self,
#                  num_classes,
#                  norm_cfg: Optional[dict] = dict(type='BN'),
#                  act_cfg=dict(type='ReLU'),
#                  init_cfg=None
#                  ):
#         super(ASPP, self).__init__(init_cfg)
#
#         self.conv_1x1_1 = ConvModule(512, 256, kernel_size=1,norm_cfg=None,act_cfg=None)
#         _,self.bn_conv_1x1_1 = build_norm_layer(norm_cfg,256)
#
#         self.conv_3x3_1 = ConvModule(512, 256, kernel_size=3, stride=1, padding=6, dilation=6,
#                                      norm_cfg=None,act_cfg=None)
#         _,self.bn_conv_3x3_1 = build_norm_layer(norm_cfg,256)
#
#         self.conv_3x3_2 = ConvModule(512, 256, kernel_size=3, stride=1, padding=12, dilation=12,
#                                      norm_cfg=None,act_cfg=None)
#         _,self.bn_conv_3x3_2 = build_norm_layer(norm_cfg,256)
#
#         self.conv_3x3_3 = ConvModule(512, 256, kernel_size=3, stride=1, padding=18, dilation=18,
#                                      norm_cfg=None,act_cfg=None)
#         _,self.bn_conv_3x3_3 = build_norm_layer(norm_cfg,256)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.conv_1x1_2 = ConvModule(512, 256, kernel_size=1,norm_cfg=None,act_cfg=None)
#         _,self.bn_conv_1x1_2 = build_norm_layer(norm_cfg,256)
#
#         self.conv_1x1_3 = ConvModule(1280, 256, kernel_size=1) # (1280 = 5*256)
#         _,self.bn_conv_1x1_3 = build_norm_layer(norm_cfg,256)
#
#         self.conv_1x1_4 = ConvModule(256, num_classes, kernel_size=1,norm_cfg=None,act_cfg=None)
#
#     def forward(self, feature_map):
#         # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))
#
#         feature_map_h = feature_map.size()[2] # (== h/16)
#         feature_map_w = feature_map.size()[3] # (== w/16)
#
#         out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#
#         out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
#         out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))
#
#         out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
#         out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
#         out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))
#
#         return out
#
class ASPP_Bottleneck(BaseModule):
    def __init__(self,
                 num_classes,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None
                 ):
        super(ASPP_Bottleneck, self).__init__(init_cfg)

        self.conv_1x1_1 = ConvModule(4*512, 256, kernel_size=1,norm_cfg=None,act_cfg=None)
        _,self.bn_conv_1x1_1 = build_norm_layer(norm_cfg,256)

        self.conv_3x3_1 = ConvModule(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6,
                                     norm_cfg=None,act_cfg=None)
        _,self.bn_conv_3x3_1 = build_norm_layer(norm_cfg,256)

        self.conv_3x3_2 = ConvModule(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12,
                                     norm_cfg=None,act_cfg=None)
        _,self.bn_conv_3x3_2 = build_norm_layer(norm_cfg,256)

        self.conv_3x3_3 = ConvModule(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18,
                                     norm_cfg=None,act_cfg=None)
        _,self.bn_conv_3x3_3 = build_norm_layer(norm_cfg,256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = ConvModule(4*512, 256, kernel_size=1,norm_cfg=None,act_cfg=None)
        _,self.bn_conv_1x1_2 = build_norm_layer(norm_cfg,256)

        self.conv_1x1_3 = ConvModule(1280, 256, kernel_size=1,norm_cfg=None,act_cfg=None) # (1280 = 5*256)
        _,self.bn_conv_1x1_3 = build_norm_layer(norm_cfg,256)

        self.conv_1x1_4 = ConvModule(256, num_classes, kernel_size=1,norm_cfg=None,act_cfg=None)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 4*512, h/16, w/16))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

        return out

'''----------------------------------------------------Net------------------------------------------------------------------------------------'''

@MODELS.register_module()
class DeepLabV3(BaseModule):
    def __init__(self,
                 norm_eval = False,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg = [dict(type='Kaiming', layer='Conv2d',
                                                  a=math.sqrt(5),
                                                  distribution='uniform',
                                                  mode='fan_in',
                                                  nonlinearity='leaky_relu'),
                                             dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])]  # 初始化配置字典
                 ):
        super(DeepLabV3, self).__init__(init_cfg)

        self.norm_eval = norm_eval

        # self.model_id = model_id
        # self.project_dir = project_dir
        # self.create_model_dirs()

        self.resnet = resnet50() # NOTE! specify the type of ResNet here
        #self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        # self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        out = []

        # h = x.size()[2]
        # w = x.size()[3]

        aux_feature,feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        out.append(aux_feature)
        out.append(feature_map)
        # output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        # output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        # out.append(output)
        # print(output.shape)

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
    x= torch.randn(2,3,512,512).cuda()
    mo = DeepLabV3()
    print(mo)
    # models =DeepLabV3(num_classes=32).cuda()
    # y = models(x)
    # print(models(x).shape)

