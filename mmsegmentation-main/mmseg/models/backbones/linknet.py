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

nonlinearity = partial(F.relu, inplace=True)
class ResidualBlock(BaseModule):
    # 实现子module：Residual Block
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 shortcut=None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(ResidualBlock, self).__init__(init_cfg)
        # self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_ch)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = ConvModule(in_ch, out_ch, 3, stride, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv2 = ConvModule(out_ch, out_ch, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=None)
        self.downsample = shortcut

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)

def BasicBlock(in_ch, out_ch, stride):
    return nn.Sequential(
        ConvModule(in_ch, out_ch, 3, stride, padding=1, bias=False,norm_cfg=None,act_cfg=None),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),  # inplace = True原地操作,节省显存
        ConvModule(out_ch, out_ch, 3, stride=1, padding=1, bias=False,norm_cfg=None,act_cfg=None),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class DecoderBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 stride=2,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(DecoderBlock, self).__init__(init_cfg)
        out_pad = 0 if stride == 1 else 1

        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.norm1 = nn.BatchNorm2d(in_channels // 4)
        # self.relu1 =
        self.conv1 = ConvModule(in_channels,in_channels//4,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=stride, padding=1,
                                          output_padding=out_pad)
        _,self.norm2 = build_norm_layer(norm_cfg,in_channels//4)
        self.relu2 = build_activation_layer(act_cfg)

        # self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nonlinearity
        self.conv3 = ConvModule(in_channels//4,n_filters,1,norm_cfg=norm_cfg,act_cfg=act_cfg)


    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

class LinkNet18(BaseModule):
    def __init__(self,
                 num_classes=1,
                 Pretrained = False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(LinkNet18, self).__init__(init_cfg)

        filters = [64, 128, 256, 512]

        resnet = models.resnet18(pretrained=Pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock(512) # 该模块 默认加载

        self.decoder4 = DecoderBlock(filters[3], filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = DecoderBlock(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = DecoderBlock(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = DecoderBlock(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = ConvModule(32, num_classes, 3, padding=1,norm_cfg=None,act_cfg=None)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)

class LinkNet34_src(nn.Module):
    # 实现主module:ResNet34 09.15已经校验 和源码 linknet相一致，
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_src, self).__init__()
        # layers = [64,128,256,512]
        # layers = filters = [32, 64, 128, 256]
        layers = filters = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 编码器
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)

        # 连接器
        # 解码器
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = None
        # 判断是否使用降采样 维度增加
        if not in_ch == out_ch:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.init_block(x)  # B * 64*256*256
        # write_feature_map(x[2], layer_name='x_src')
        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)


@MODELS.register_module()
class LinkNet34(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=32,
                 Pretrained = False,
                 norm_eval: bool = False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = [dict(type='Kaiming',layer='Conv2d',
                                                 a=math.sqrt(5),
                                                 distribution='uniform',
                                                 mode='fan_in',
                                                 nonlinearity='leaky_relu'),
                                             dict(type='Constant',val=1,layer=['_BatchNorm', 'GroupNorm'])]  # 初始化配置字典
                 ):
        super(LinkNet34, self).__init__(init_cfg)
        self.norm_eval=norm_eval

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=Pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock(512) # 该模块 默认加载

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
        #    self.last_activation = torch.nn.Sigmoid()
        # else:
        #    self.last_activation = torch.nn.Softmax(dim=1)


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

        # Decoder
        d4 = self.decoder4(e4) + e3
        output.append(d4)               # index=0
        d3 = self.decoder3(d4) + e2
        output.append(d3)               # index=1
        d2 = self.decoder2(d3) + e1
        output.append(d2)               # index=2
        d1 = self.decoder1(d2)
        output.append(d1)               # index=3


        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        output.append(out)              # index=4

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


@MODELS.register_module()
class LinkNet50(BaseModule):
    def __init__(self,
                 base_channels=32,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = [dict(type='Kaiming', layer='Conv2d',
                                                  a=math.sqrt(5),
                                                  distribution='uniform',
                                                  mode='fan_in',
                                                  nonlinearity='leaky_relu'),
                                             dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])]  # 初始化配置字典
                 ):
        super(LinkNet50, self).__init__(init_cfg)

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = DecoderBlock(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = DecoderBlock(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = DecoderBlock(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = ConvModule(32, base_channels, 3, padding=1,norm_cfg=None,act_cfg=None)


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
        # e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        output.append(d4)                   # index=0
        d3 = self.decoder3(d4) + e2
        output.append(d3)                   # index=1
        d2 = self.decoder2(d3) + e1
        output.append(d2)                   # index=2
        d1 = self.decoder1(d2)
        output.append(d1)                   # index=3
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        output.append(out)                   # index=4


        return self.last_activation(out)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()



class LinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
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
        # e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)

if __name__ == '__main__':
    x = torch.randn(1,3,64,64).cuda()
    net =LinkNet34(base_channels=32).cuda()  # Trainable params: 11,548,737
    # net = LinkNet34(num_classes=1) # Trainable params: 21,656,897
    out_list = net(x)