import torch
import torch.nn as nn
import math
from functools import partial
from torch.nn import functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from typing import Optional
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm


'''----------------------------------------------------------------------'''
class ChannelAttention(BaseModule):
    def __init__(self,
                 in_planes:int,
                 out_planes:int,
                 ratio:int=2,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(ChannelAttention, self).__init__(init_cfg)
        self.conv = ConvModule(in_planes, out_planes, 1, bias=False,norm_cfg=None,act_cfg=None)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc11 = ConvModule(out_planes, out_planes // ratio, 1, bias=False,norm_cfg=None,act_cfg=None)
        self.fc12 = ConvModule(out_planes // ratio, out_planes, 1, bias=False,norm_cfg=None,act_cfg=None)

        self.fc21 = ConvModule(out_planes, out_planes // ratio, 1, bias=False,norm_cfg=None,act_cfg=None)
        self.fc22 = ConvModule(out_planes // ratio, out_planes, 1, bias=False,norm_cfg=None,act_cfg=None)
        self.relu1 = build_activation_layer(act_cfg)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        del avg_out, max_out
        return x * self.sigmoid(out)
'''----------------------------------------------------------------------'''
nonlinearity = partial(F.relu, inplace=True)
tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)
class ResidualBlock(BaseModule):
    # 实现子module：Residual Block
    def __init__(self,
                 in_ch:int,
                 out_ch:int,
                 stride:int=1,
                 shortcut=None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(ResidualBlock, self).__init__(init_cfg)
        self.conv1 = ConvModule(in_ch,out_ch,3,stride,padding=1,bias=False,norm_cfg=norm_cfg,act_cfg=None)
        self.conv2 = ConvModule(out_ch,out_ch,3,1,1,bias=False,norm_cfg=norm_cfg,act_cfg=None)

        self.downsample = shortcut

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)

        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)

class connecter(BaseModule):
    # 连接器
    def __init__(self,
                 in_ch,
                 out_ch,
                 scale_factor=0.5,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(connecter, self).__init__(init_cfg)
        self.downsample = partial(F.interpolate, scale_factor=scale_factor, mode='area', recompute_scale_factor=True)
        # mode(str)：用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'。默认：'nearest'
        # /home/wyc/software/anconda3/envs/pytroch_test/lib/python3.9/site-packages/torch/nn/functional.py:3502:
        # UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries,
        # and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior,
        # please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.

        if not in_ch == out_ch:
            shortcut = ConvModule(in_ch, out_ch,1,1,norm_cfg=norm_cfg,act_cfg=None)
        else:
            shortcut = None
        self.connect_conv = ResidualBlock(in_ch, out_ch, shortcut=shortcut,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x):
        x = self.downsample(x)
        x = self.connect_conv(x)
        return x

class encoder_i(BaseModule):
    def __init__(self,
                 scale_factor:int,
                 in_c:int=64,
                 res_layers:int=64,
                 num_res_blocks:int=3,
                 stride:int=1,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(encoder_i, self).__init__(init_cfg)
        # 编码器
        self.resnet_i = self.make_layer(in_c, res_layers, num_res_blocks, stride)
        self.connect_i = connecter(3, res_layers, scale_factor=scale_factor,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x_input, res_input):
        out1 = self.resnet_i(res_input)
        out2 = self.connect_i(x_input)
        out = torch.cat((out1, out2), 1)
        return out

    def make_layer(self,
                   in_ch:int,
                   out_ch:int,
                   block_num:int,
                   stride=1,
                   ):
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

class initblock_plus4(BaseModule):
    # 激活函数用的relu bing xing
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias = False,
                 strip=9,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(initblock_plus4, self).__init__(init_cfg)

        self.conv_a0 = nn.Sequential(ConvModule(in_ch,32,kernel_size,stride,padding,dilation, bias=False,norm_cfg=norm_cfg,act_cfg=None),
                                     nn.ELU(inplace=True)
                                     )

        self.multi_conv1 = ConvModule(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2),norm_cfg=None,act_cfg=None)
        self.multi_conv2 = ConvModule(in_ch, 8, (strip, 1), stride=stride,padding=(strip//2, 0),norm_cfg=None,act_cfg=None)
        self.multi_conv3 = ConvModule(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2),norm_cfg=None,act_cfg=None)
        self.multi_conv4 = ConvModule(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2),norm_cfg=None,act_cfg=None)

        self.channel_concern = nn.Sequential(ConvModule(32, 32, 1, 1, padding=0,dilation=dilation, bias=bias,
                                                        norm_cfg=norm_cfg,act_cfg=None),
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
class DecoderBlock_v4fix(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 in_p=True,
                 strip=9,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(DecoderBlock_v4fix, self).__init__(init_cfg)
        out_pad = 1 if in_p else 0
        stride = 2 if in_p else 1

        self.cbr1 = ConvModule(in_channels,in_channels//4,1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.cbr2 = ConvModule(in_channels, in_channels // 2, 1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv1 = ConvModule(in_channels // 4, in_channels // 4, (1, strip), padding=(0, strip//2),norm_cfg=None,act_cfg=None)
        self.deconv2 = ConvModule( in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0),norm_cfg=None,act_cfg=None)
        self.deconv3 = ConvModule(in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0),norm_cfg=None,act_cfg=None)
        self.deconv4 = ConvModule(in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0),norm_cfg=None,act_cfg=None)

        self.cbr3_1 = ConvModule(in_channels // 4 + in_channels // 2, in_channels // 4, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.cbr3_2 = ConvModule(in_channels // 4 + in_channels // 2, in_channels // 4, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.cbr3_3 = ConvModule(in_channels // 4 + in_channels // 2, in_channels // 4, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.cbr3_4 = ConvModule(in_channels // 4 + in_channels // 2, in_channels // 4, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)


        self.deconvbr = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                                                         3, stride=stride, padding=1, output_padding=out_pad),
                                      nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
                                      nn.ReLU(inplace=True), )

        self.conv3 = ConvModule(in_channels // 4 + in_channels // 4, n_filters, 1,norm_cfg=norm_cfg)
        _,self.bn3 = build_norm_layer(norm_cfg,n_filters)
        self.relu3 = build_activation_layer(act_cfg)


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
@MODELS.register_module()
class MSMDFF_Net_v3_plus(BaseModule):
    # 2023.07.11 在整个网络中使用 旋转特征图的方向带状卷积方式
    #
    def __init__(self,
                 in_channels=3,
                 base_channels=32,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = [dict(type='Kaiming', layer='Conv2d',
                                                  a=math.sqrt(5),
                                                  distribution='uniform',
                                                  mode='fan_in',
                                                  nonlinearity='leaky_relu'),
                                             dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])],  # 初始化配置字典
                 norm_eval=False

                 ):
        super(MSMDFF_Net_v3_plus, self).__init__(init_cfg)
        self.norm_eval = norm_eval

        layers = [64, 128, 256, 512]
        self.init_block = initblock_plus4(in_channels,64,stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)

        # 编码器 512 256 128 64
        self.encoder1 = encoder_i(0.5, in_c=layers[0], res_layers=layers[0], num_res_blocks=3, stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.encoder2 = encoder_i(0.25, in_c=layers[1], res_layers=layers[1], num_res_blocks=4, stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.max_pool2 = nn.MaxPool2d(3, 2, 1)
        self.encoder3 = encoder_i(0.125, in_c=layers[2], res_layers=layers[2], num_res_blocks=6, stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.max_pool3 = nn.MaxPool2d(3, 2, 1)
        self.encoder4 = encoder_i(0.0625, in_c=layers[3], res_layers=layers[3], num_res_blocks=3, stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # link
        self.c5 = connecter(1024, 512, scale_factor=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.c4 = connecter(512, 256, scale_factor=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.c3 = connecter(256, 128, scale_factor=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.c2 = connecter(128, 64, scale_factor=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # 解码器
        self.decoder4 = DecoderBlock_v4fix(layers[3], layers[2],in_p=True,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = DecoderBlock_v4fix(layers[2], layers[1],in_p=True,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = DecoderBlock_v4fix(layers[1], layers[0],in_p=True,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = DecoderBlock_v4fix(layers[0], layers[0],in_p=True,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.finalconv2 = ConvModule(layers[0], 32, 3, padding=1,norm_cfg=None,act_cfg=None)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = ConvModule(32, base_channels, 3, padding=1,norm_cfg=None,act_cfg=None)


    def forward(self, x):
        out = []
        x1 = self.init_block(x)

        e1 = self.encoder1(x, x1)  # 128**h/2*w/2
        # print(e1.shape)
        e2 = self.encoder2(x, e1)  # 256**h/4*w/4
        e3 = self.encoder3(x, e2)  # 512**h/8*w/8
        e4 = self.encoder4(x, e3)  # 1024*h/16*w/16

        c4 = self.c5(e4)            # 512*h/16*w/16
        out.append(c4)                          # index=0
        # Decoder
        d4 = self.decoder4(c4) + self.c4(e3)   # 256+256*h/8*w/8
        # print(d4.shape)
        out.append(self.decoder4(c4))           # index=1
        d3 = self.decoder3(d4) + self.c3(e2)   # 128+128*h/4*w/4
        # print(d3.shape)
        out.append(self.decoder3(d4))           # index=2
        d2 = self.decoder2(d3) + self.c2(e1)   # 64+64*h/2*w/2
        # print(d2.shape)

        out.append(self.decoder2(d3))           # index=3
        d1 = self.decoder1(d2)    # 64*h*w
        # print(d1.shape)

        output = self.finalconv2(d1)
        output = self.finalrelu2(output)
        output = self.finalconv3(output)
        out.append(output)                         # index=4

        # torch.cuda.empty_cache()
        # return torch.sigmoid(out)

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
'''
中间跳跃层的代价十分大，后面可考虑是否使用中间跳跃层
此外，反卷积的那个 带状卷积 再次尝试一下。
'''

if __name__ == '__main__':
    x = torch.randn(2,3,64,64).cuda()
    moudle = MSMDFF_Net_v3_plus(3,64).cuda()
    print(moudle(x).shape)