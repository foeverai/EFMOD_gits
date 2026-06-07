# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/4 上午10:15
# @Author  : 王玉川 uestc
# @File    : Multi_scale_LinkNet34.py
# @Description :

import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
# from networks.A11_MSMDFFNet.init_block.multi_D_initblock import initblock_plus4
# from networks.A11_MSMDFFNet.encoder.MSENCODER import encoder_i,connecter,nonlinearity
# from networks.A11_MSMDFFNet.model.decoder import DecoderBlock_v4fix as F_DecoderBlock_v4fix

# https://blog.51cto.com/u_15906550/5921646 旋转特征图
'''----------------------------------------------------------------------'''
nonlinearity = partial(F.relu, inplace=True)
tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)
class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)


class connecter(nn.Module):
    # 连接器
    def __init__(self, in_ch, out_ch, scale_factor=0.5):
        super(connecter, self).__init__()
        self.downsample = partial(F.interpolate, scale_factor=scale_factor, mode='area', recompute_scale_factor=True)
        # mode(str)：用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'。默认：'nearest'
        # /home/wyc/software/anconda3/envs/pytroch_test/lib/python3.9/site-packages/torch/nn/functional.py:3502:
        # UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries,
        # and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior,
        # please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.

        if not in_ch == out_ch:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        else:
            shortcut = None
        self.connect_conv = ResidualBlock(in_ch, out_ch, shortcut=shortcut)

    def forward(self, x):
        x = self.downsample(x)
        x = self.connect_conv(x)
        return x

class encoder_i(nn.Module):
    def __init__(self, scale_factor, in_c=64, res_layers=64, num_res_blocks=3, stride=1):
        super(encoder_i, self).__init__()
        # 编码器
        self.resnet_i = self.make_layer(in_c, res_layers, num_res_blocks, stride)
        self.connect_i = connecter(3, res_layers, scale_factor=scale_factor)

    def forward(self, x_input, res_input):
        out1 = self.resnet_i(res_input)
        out2 = self.connect_i(x_input)
        out = torch.cat((out1, out2), 1)
        return out

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = None
        # 判断是否使用降采样 维度增加
        if not in_ch == out_ch or not stride == 1:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)
'''----------------------------------------------------------------------'''

class initblock_plus4(nn.Module):
    # 激活函数用的relu bing xing
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1,dilation=1,bias = False,strip=9):
        super(initblock_plus4, self).__init__()



        self.conv_a0 = nn.Sequential(nn.Conv2d(in_ch, 32, kernel_size, stride, padding=padding,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(inplace=True)
                                      )


        self.multi_conv1 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))
        self.multi_conv2 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride,padding=(strip//2, 0))
        self.multi_conv3 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))
        self.multi_conv4 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))

        self.channel_concern = nn.Sequential(nn.Conv2d(32, 32, 1, 1, padding=0,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(inplace=True)
                                      )

        self.angle = [0,45,90,135,180]
    def forward(self, x):


        x1 = self.multi_conv1(x)
        x2 = self.multi_conv2(x)
        x3 = self.conv_a0(x)
        x4 = self.multi_conv3(tensor_rotate(x,self.angle[1]))
        x5 = self.multi_conv4(tensor_rotate(x,self.angle[3]))

        out = torch.cat((x1,
                         x2,
                         tensor_rotate(x4,-self.angle[1]),
                         tensor_rotate(x5,-self.angle[3]),
                         ), 1)
        out = torch.cat((self.channel_concern(out),
                         x3),1)
        return out

'''---------------------------------------------------------------------------------------------------'''
class DecoderBlock_v4fix(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, in_p=True,strip=9):
        super(DecoderBlock_v4fix, self).__init__()
        out_pad = 1 if in_p else 0
        stride = 2 if in_p else 1

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.cbr2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True), )

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (1, strip), padding=(0, strip//2)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0)
        )

        self.cbr3_1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_3 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_4 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.deconvbr = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                                                         3, stride=stride, padding=1, output_padding=out_pad),
                                      nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
                                      nn.ReLU(inplace=True), )

        self.conv3 = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()


    def forward(self, x, inp=False):
        x01 = self.cbr1(x)

        x02 = self.cbr2(x)

        x1 = self.deconv1(x01)
        x2 = self.deconv2(x01)
        x3 = tensor_rotate(self.deconv3(tensor_rotate(x01, 45)), -45)
        x4 = tensor_rotate(self.deconv4(tensor_rotate(x01,135)),-135)

        x1 = self.cbr3_1(torch.cat((x1, x02), 1))
        x2 = self.cbr3_2(torch.cat((x2, x02), 1))
        x3 = self.cbr3_3(torch.cat((x3, x02), 1))
        x4 = self.cbr3_4(torch.cat((x4, x02), 1))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.deconvbr(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class MSMDFF_Net_v3_plus(nn.Module):
    # 2023.07.11 在整个网络中使用 旋转特征图的方向带状卷积方式
    #
    def __init__(self, in_c=3, num_classes=1):
        super(MSMDFF_Net_v3_plus, self).__init__()

        layers = [64, 128, 256, 512]
        self.init_block = initblock_plus4(3,64,stride=2)

        # 编码器 512 256 128 64
        self.encoder1 = encoder_i(0.5, in_c=layers[0], res_layers=layers[0], num_res_blocks=3, stride=1)
        # self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.encoder2 = encoder_i(0.25, in_c=layers[1], res_layers=layers[1], num_res_blocks=4, stride=2)
        # self.max_pool2 = nn.MaxPool2d(3, 2, 1)
        self.encoder3 = encoder_i(0.125, in_c=layers[2], res_layers=layers[2], num_res_blocks=6, stride=2)
        # self.max_pool3 = nn.MaxPool2d(3, 2, 1)
        self.encoder4 = encoder_i(0.0625, in_c=layers[3], res_layers=layers[3], num_res_blocks=3, stride=2)
        # link
        self.c5 = connecter(1024, 512, scale_factor=1)
        self.c4 = connecter(512, 256, scale_factor=1)
        self.c3 = connecter(256, 128, scale_factor=1)
        self.c2 = connecter(128, 64, scale_factor=1)
        # 解码器
        self.decoder4 = DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        # self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(layers[0], 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, x):
        x1 = self.init_block(x)

        e1 = self.encoder1(x, x1)  # 128*256*256
        e2 = self.encoder2(x, e1)  # 256*128*128
        e3 = self.encoder3(x, e2)  # 512*64*64
        e4 = self.encoder4(x, e3)  # 512*32*32

        c4 = self.c5(e4)
        # Decoder
        d4 = self.decoder4(c4) + self.c4(e3)   # 256*64*64
        d3 = self.decoder3(d4) + self.c3(e2)   # 128*128*128
        d2 = self.decoder2(d3) + self.c2(e1)   # 64*256*256
        d1 = self.decoder1(d2)    # 64*256*256

        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)
        out = self.finalconv2(d1)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # torch.cuda.empty_cache()
        return torch.sigmoid(out)
'''
中间跳跃层的代价十分大，后面可考虑是否使用中间跳跃层
此外，反卷积的那个 带状卷积 再次尝试一下。
'''

if __name__ == '__main__':
    x = torch.randn(2,3,64,64).cuda()
    moudle = MSMDFF_Net_v3_plus(3,64).cuda()
    print(moudle(x).shape)