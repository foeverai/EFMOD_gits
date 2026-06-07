import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
from mmseg.registry import MODELS
from mmcv.cnn.bricks import DropPath
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from functools import partial
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmseg.models.utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)
'''-----------------------------------------------------------------------------------------------------------------'''
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


"""-----------------------------------------------------------------------------------------------------------------"""
class stem_block(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 dilation=1,
                 bias = False,
                 strip=9,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        self.conv_a0 = nn.Sequential(ConvModule(in_ch,out_ch,3,1,1,norm_cfg=norm_cfg,act_cfg=None),
                                     nn.ELU(inplace=True)
                                     )

        self.multi_conv1 = ConvModule(out_ch, 8, (1, strip), stride=stride, padding=(0, strip // 2), norm_cfg=None,
                                      act_cfg=None)
        self.multi_conv2 = ConvModule(out_ch, 8, (strip, 1), stride=stride, padding=(strip // 2, 0), norm_cfg=None,
                                      act_cfg=None)
        self.multi_conv3 = ConvModule(out_ch, 8, (1, strip), stride=stride, padding=(0, strip // 2), norm_cfg=None,
                                      act_cfg=None)
        self.multi_conv4 = ConvModule(out_ch, 8, (1, strip), stride=stride, padding=(0, strip // 2), norm_cfg=None,
                                      act_cfg=None)

        self.channel_concern = nn.Sequential(ConvModule(out_ch, out_ch, 1, 1, padding=0,dilation=dilation, bias=bias,
                                                        norm_cfg=norm_cfg,act_cfg=None),
                                      nn.ELU(inplace=True)
                                      )

        self.angle = [0,45,90,135,180]
    def forward(self, x):
        x = self.conv_a0(x)

        x1 = self.multi_conv1(x)
        x2 = self.multi_conv2(x)
        x4 = self.multi_conv3(tensor_rotate(x,self.angle[1]))
        x5 = self.multi_conv4(tensor_rotate(x,self.angle[3]))

        out = torch.cat((x1,
                         x2,
                         tensor_rotate(x4,-self.angle[1]),
                         tensor_rotate(x5,-self.angle[3]),
                         ), 1)
        out = self.channel_concern(out)+x
        return out

'''----------------------------------------------------Encoder--------------------------------------------------------'''
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



class Encoder_block(BaseModule):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.down_conv=nn.Sequential(nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg))
        self.hv_conv = nn.Sequential(ConvModule(out_channels,out_channels,(1,3),1,(0,1),norm_cfg=None,act_cfg=None),
                      ConvModule(out_channels,out_channels,(1,3),1,(0,1),norm_cfg=norm_cfg,act_cfg=None)
                      )
        self.act = build_activation_layer(act_cfg)
        self.conv_1x1 = ConvModule(out_channels*2,out_channels,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self,x):
        x_1= self.down_conv(x)
        x_2 = self.hv_conv(x_1)
        x = self.act(torch.cat((x_1,x_2),1))

        return self.conv_1x1(x)

"""----------------------------------------------------ASPP_aug--------------------------------------------------------"""
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

'''--------------------------------------------------downsample----------------------------------------------------------'''
class DownSamplingLayer(BaseModule):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.down_conv(x)

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

'''------------------------------------------------------Net-------------------------------------------------------------'''

@MODELS.register_module()
class RMFMNet_ablation_att(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=32,
                 bilinear=False,
                 norm_eval: bool = False,  # 是否在评估模式下使用归一化，默认为false
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = [dict(type='Kaiming',layer='Conv2d',
                                                 a=math.sqrt(5),
                                                 distribution='uniform',
                                                 mode='fan_in',
                                                 nonlinearity='leaky_relu'),
                                             dict(type='Constant',val=1,layer=['_BatchNorm', 'GroupNorm'])]  # 初始化配置字典
                 ):
        super().__init__(init_cfg)
        self.input_channel = in_channels
        self.class_num = base_channels
        self.norm_eval = norm_eval

        out_channels = [32, 64, 128, 256, 512]

        self.conv1 = stem_block(self.input_channel,out_channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)  #(bs,c,h,w)→(bs,32,h,w)


        self.conv2 = nn.Sequential(
            # (bs,32,h,w)→(bs,32,h/2,w/2)
            Encoder_block(out_channels[0],out_channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )  # (bs,32,h,w)→(bs,64,h/2,w/2)
        self.conv1x1_1 = ConvModule(out_channels[1]*2,out_channels[1],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv3 = nn.Sequential(
            # (bs,64,h/2,w/2)→(bs,64,h/4,w/4)
            Encoder_block(out_channels[1],out_channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )   # (bs,64,h/4,w/4)→(bs,128,h/4,w/4)
        self.conv1x1_2 = ConvModule(out_channels[2]*2,out_channels[2],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.conv4 = nn.Sequential(
            # (bs,128,h/4,w/4)→(bs,128,h/8,w/8)
            Encoder_block(out_channels[2],out_channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )   # (bs,128,h/4,w/4)→(bs,256,h/8,w/8)
        self.conv1x1_3 = ConvModule(out_channels[3]*2,out_channels[3],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.conv5 = nn.Sequential(
            # (bs,256,h/8,w/8)→(bs,256,h/16,w/16)
            Encoder_block(out_channels[3],out_channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )   # (bs,256,h/8,w/8)→(bs,512,h/16,w/16).
        self.conv1x1_4 = ConvModule(out_channels[4]*2,out_channels[4],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv5 = ASPP(out_channels[4],[1,6,12,18],out_channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.connect1 = connecter(3,64,0.5,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.connect2 = connecter(3,128,0.25,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.connect3 = connecter(3,256,0.125,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.connect4 = connecter(3,512,0.0625,norm_cfg=norm_cfg,act_cfg=act_cfg)

        factor = 2 if bilinear else 1  # 默认bilinear==False
        self.up_1 = Up(512, 256 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_2 = Up(256, 128 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_3 = Up(128, 64 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_4 = Up(64, 32 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.final_conv = ConvModule(out_channels[0], self.class_num, kernel_size=1, stride=1,norm_cfg=None,act_cfg=None)  # (bs,32,h,w)→(bs,num_classes,h,w)


    def forward(self, x):
        x = x.contiguous()
        y = x.clone()
        out = []
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        connect1 = self.connect1(y)
        conv2 = self.conv1x1_1(torch.cat((conv2 ,connect1),1))

        conv3 = self.conv3(conv2)
        connect2 = self.connect2(y)
        conv3 = self.conv1x1_2(torch.cat((conv3,connect2),1))

        conv4 = self.conv4(conv3)
        connect3 = self.connect3(y)
        conv4 = self.conv1x1_3(torch.cat((conv4,connect3),1))


        conv5 = self.conv5(conv4)
        connect4 = self.connect4(y)
        conv5 = self.conv1x1_4(torch.cat((conv5,connect4),1))
        # (bs,512,h/8,w/8)→(bs,1024,h/16,w/16).

        deconv5 = self.deconv5(conv5)
        out.append(deconv5)                                     # index=0

        decoder1 = self.up_1(deconv5, conv4)
        out.append(decoder1)  # index=1
        decoder2 = self.up_2(decoder1, conv3)
        out.append(decoder2)  # index=2
        decoder3 = self.up_3(decoder2, conv2)
        out.append(decoder3)  # index=3
        decoder4 = self.up_4(decoder3, conv1)

        output = self.final_conv(decoder4)
        out.append(output)  # index=4

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
    x = torch.randn(2,3,32,32).cuda()
    mo = RMFMNet_ablation_att()
    print(mo)
    # moduel = RMFMNet_ablation_att(3).cuda()
    # print(moduel(x)[4].shape)


