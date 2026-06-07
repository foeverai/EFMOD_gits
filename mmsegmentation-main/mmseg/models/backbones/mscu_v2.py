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


# https://blog.51cto.com/u_15906550/5921646 旋转特征图
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
                 strip=3,
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

'''-----------------------------------ASPP------------------------------------------------------------'''
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
'''---------------------------------------------------------------------------------------------------'''
@MODELS.register_module()
class MSCU_Net_v2(BaseModule):
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
        super(MSCU_Net_v2, self).__init__(init_cfg)
        self.norm_eval = norm_eval
        self.out_channels = base_channels

        channels = [32, 64, 128, 256]
        self.init_block = initblock_plus4(in_channels,32,stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv_in = nn.Sequential(
            ACBlock(channels[0]*2, channels[0], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[0], channels[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        # 编码器 512 256 128 64
        self.encoder1 = encoder_i(0.5, in_c=channels[0], res_layers=channels[0], num_res_blocks=3, stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.encoder2 = encoder_i(0.25, in_c=channels[1], res_layers=channels[1], num_res_blocks=4, stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.max_pool2 = nn.MaxPool2d(3, 2, 1)
        self.encoder3 = encoder_i(0.125, in_c=channels[2], res_layers=channels[2], num_res_blocks=6, stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.max_pool3 = nn.MaxPool2d(3, 2, 1)
        self.encoder4 = encoder_i(0.0625, in_c=channels[3], res_layers=channels[3], num_res_blocks=3, stride=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        #
        self.conv12 = nn.Sequential(
            ACBlock(channels[0], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv13 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv14 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        #
        self.conv22 = nn.Sequential(
            ACBlock(channels[0], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        #
        self.conv32 = nn.Sequential(
            ACBlock(channels[0], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        # ca
        self.skblock4 = ChannelAttention(channels[3]*5, channels[3]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock3 = ChannelAttention(channels[2]*5, channels[2]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock2 = ChannelAttention(channels[1]*5, channels[1]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock1 = ChannelAttention(channels[0]*5, channels[0]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)


        # 解码器
        # self.up4 = Up(channels[3]*2, channels[3] // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up4 = nn.ConvTranspose2d(channels[3]*2, channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.up0 = nn.ConvTranspose2d(channels[0], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.decoder4 = nn.Sequential(
            ConvModule(channels[3]*2,channels[3],3,1,1,norm_cfg=None,act_cfg=None),
            ACBlock(channels[3],channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.decoder3 = nn.Sequential(
            ConvModule(channels[3], channels[2], 3, 1, 1, norm_cfg=None, act_cfg=None),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.decoder2 = nn.Sequential(
            ConvModule(channels[2], channels[1], 3, 1, 1, norm_cfg=None, act_cfg=None),
            ACBlock(channels[1], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.decoder1 = nn.Sequential(
            ConvModule(channels[1], self.out_channels, 3, 1, 1, norm_cfg=None, act_cfg=None),
            ACBlock(self.out_channels, self.out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

    def forward(self, x):
        out = []
        x1 = self.init_block(x)         # (bs,32,h/2,w/2)
        x1 = self.conv_in(x1)
        # print(x1.shape)

        e1 = self.encoder1(x, x1)       # (bs,64,h/2,w/2)
        # print(e1.shape)
        e2 = self.encoder2(x, e1)       # (bs,128,h/4,w/4)
        e3 = self.encoder3(x, e2)       # (bs,256,h/8,w/8)
        e4 = self.encoder4(x, e3)       # (bs,512,h/16,w/16)
        # e4 = self.spp(e4)               # (bs,512,h/16,w/16)
        out.append(e4)                  # index=0
        # decoder_data1
        d_up1 = self.up4(e4)            # (bs,256,h/8，w/8)
        e0_in = self.conv12(x1)         # (bs,64,h/4,w/4)
        e0_in = self.conv13(e0_in)      # (bs,128,h/4,2/4)
        e0_in = self.conv14(e0_in)      # (bs,256,h/8,w/8)

        e1_in = self.conv13(e1)         # (bs,128,h/4,w/4)
        e1_in = self.conv14(e1_in)      # (bs,256,h/8,w/8)

        e2_in = self.conv14(e2)         # (bs,256,h/8,w/8)

        decoder_in = torch.cat((e0_in,e1_in,e2_in,e3,d_up1),dim=1)  # (bs,256*5,h/8,w/8)
        decoder4 = self.decoder4(self.skblock4(decoder_in))        # (bs,256,h/8,w/8)
        del d_up1,e0_in,e1_in,e2_in,decoder_in
        out.append(decoder4)            # index=1

        # decoder_data2
        d_up2 = self.up4(e4)            # (bs,256,h/8,w/8)
        d_up2 = self.up3(d_up2)         # (bs,128,h/4,w/4)

        e01_in = self.conv22(x1)        # (bs,64,h/2,w/2)
        e01_in = self.conv23(e01_in)    # (bs,128,h/4,w/4)

        e11_in = self.conv23(e1)        # (bs,128,h/4,w/4)

        decoder_in1 = self.up3(decoder4) # (bs,128,h/4,w/4)

        decoder_in = torch.cat((e01_in,e11_in,e2,decoder_in1,d_up2),dim=1)  # (bs,128*5,h/4,w/4)
        decoder3 = self.decoder3(self.skblock3(decoder_in))        # (bs,128,h/4,w/4)

        del d_up2,decoder_in,decoder_in1,e11_in,e01_in
        out.append(decoder3)            # index=2
        # decoder_data3
        d_up3 = self.up4(e4)            # (bs,256,h/8,w/8)
        d_up3 = self.up3(d_up3)         # (bs,128,h/4.w/4)
        d_up3 = self.up2(d_up3)         # (bs,64,h/2,w/2)

        e02_in = self.conv32(x1)        # (bs,64,h/2,w/2)

        decoder_in1 = self.up3(decoder4)           # (bs,64,h/4,w/4)
        decoder_in1 = self.up2(decoder_in1)        # (bs,64,h/2,w/2)

        decoder_in2 = self.up2(decoder3)        # (bs,64,h/2,w/2)

        decoder_in = torch.cat((e02_in,e1,decoder_in2,decoder_in1,d_up3),dim=1)
        decoder2 = self.decoder2(self.skblock2(decoder_in))     # (bs,64,h/2,w/2)
        del d_up3,e02_in,decoder_in1,decoder_in2,decoder_in
        out.append(decoder2)            # index=3
        # decoder_data4
        d_up4 = self.up4(e4)  # (bs,256,h/8,w/8)
        d_up4 = self.up3(d_up4)  # (bs,128,h/4.w/4)
        d_up4 = self.up2(d_up4)  # (bs,64,h/2,w/2)
        d_up4 = self.up1(d_up4)  # (bs,32,h,w)

        decoder_in1 = self.up3(decoder4)  # (bs,128,h/4,w/4)
        decoder_in1 = self.up2(decoder_in1)  # (bs,64,h/2,w/2)
        decoder_in1 = self.up1(decoder_in1)  # (bs,32,h/2,w/2)

        decoder_in2 = self.up2(decoder3)        # (bs,64,h/2,w/2)
        decoder_in2 = self.up1(decoder_in2)     # (bs,32,h,w)

        decoder_in3 = self.up1(decoder2)        # (bs,32,h,w)

        e1_in = self.up0(x1)                    # (bs,32,h,w)

        decoder_in = torch.cat((e1_in,decoder_in3,decoder_in2,decoder_in1,d_up4),dim=1)  # (bs,32*5,h,w)
        decoder1 = self.decoder1(self.skblock1(decoder_in))     # (bs,32,h,w)
        # print(decoder1.shape)
        out.append(decoder1)                # index=4
        del d_up4,decoder_in1,decoder_in2,decoder_in3,e1_in,decoder_in
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
    moudle = MSCU_Net_v2(3,32).cuda()
    print(moudle(x)[4].shape)