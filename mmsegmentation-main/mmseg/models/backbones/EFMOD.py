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
class MFE_Block(BaseModule):
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



class EFE(BaseModule):
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



'''------------------------------------------branch_conv-----------------------------------------------------'''
class SE(BaseModule):
    def __init__(self,
                 in_channel,
                 ratio=16,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # ? c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # ? c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y*x


class BRANCH_CONV16(BaseModule):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        # self.gn = GroupBatchnorm2d16(oup_channels, group_num=group_num,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.gn = nn.GroupNorm(num_groups=group_num,num_channels=oup_channels)
        self.cha_at = SE(oup_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)


    def forward(self, x):
        gn_x= self.gn(x)
        out = self.cha_at(gn_x)
        return out


class GroupBatchnorm2d4(BaseModule):
    def __init__(self,
                 c_num: int,
                 group_num: int = 4,
                 eps: float = 1e-10,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(GroupBatchnorm2d4, self).__init__(init_cfg)
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class BRANCH_CONV4(BaseModule):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 4,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        self.gn = nn.GroupNorm(group_num,oup_channels)
        self.cha_at = SE(oup_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)


    def forward(self, x):
        gn_x = self.gn(x)
        out = self.cha_at(gn_x)
        return out




class DepthWiseConv2d(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 kernel_size:int=3,
                 padding:int=1,
                 stride:int=1,
                 dilation:int=1,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        # self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
        #                        stride=stride, dilation=dilation, groups=dim_in)
        self.conv1 = ConvModule(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                                stride=stride, dilation=dilation, groups=dim_in,norm_cfg=None,act_cfg=None)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        # self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.conv2 = ConvModule(dim_in, dim_out, kernel_size=1, stride=1, norm_cfg=None, act_cfg=None)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last",
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MOD_mutil1(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 x:int=8,
                 y:int=8,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 conv_cfg: Optional[dict] = dict(type='Conv1d'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        c_dim_in = dim_in//4
        k_size =3
        pad =(k_size -1) // 2

        self.branch_conv = BRANCH_CONV4(c_dim_in,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.params_c = nn.Parameter(torch.Tensor(1, c_dim_in, 1, 1), requires_grad=True)
        # self.conv_c = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
        self.conv_c = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1,norm_cfg=None, act_cfg=None)
                                    )

        self.params_x = nn.Parameter(torch.Tensor(1, 1, x, 1), requires_grad=True)
        nn.init.ones_(self.params_x)
        # self.conv_x = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
        self.conv_x = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg,conv_cfg=conv_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1,norm_cfg=None,act_cfg=None,conv_cfg=conv_cfg)
                                    )
        self.params_y = nn.Parameter(torch.Tensor(1, 1, 1, y), requires_grad=True)
        nn.init.ones_(self.params_y)
        self.conv_y = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg,conv_cfg=conv_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1,norm_cfg=None,act_cfg=None,conv_cfg=conv_cfg)
                                    )

        self.dw = nn.Sequential(ConvModule(c_dim_in,c_dim_in,1,norm_cfg=None,act_cfg=act_cfg),
                                ConvModule(c_dim_in,c_dim_in,3,1,1,groups=c_dim_in,norm_cfg=None,act_cfg=None)

        )
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first',norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first',norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.ldw = nn.Sequential(ConvModule(dim_in,dim_in,3,1,1,groups=dim_in,norm_cfg=None,act_cfg=act_cfg),
                                 ConvModule(dim_in,dim_out,1,1,norm_cfg=None,act_cfg=None)
             )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        params_c = self.params_c
        x1 = x1 * self.conv_c(F.interpolate(params_c, size=x1.shape[2:4] ,mode='bilinear', align_corners=True))
        x1 = self.branch_conv(x1)

        x2 = x2.permute(0, 3, 1, 2)
        params_x = self.params_x
        x2 = x2 * self.conv_x(F.interpolate(params_x, size=x2.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.branch_conv(x2)

        x3 = x3.permute(0, 2, 1, 3)
        params_y = self.params_y
        x3 = x3 * self.conv_y(F.interpolate(params_y, size=x3.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x3 = self.branch_conv(x3)

        x4 = self.dw(x4)
        x4 = self.branch_conv(x4)

        x = torch.cat([x1,x2,x3,x4],dim=1)

        x = self.norm2(x)
        x = self.ldw(x)

        return x
'''---------------------------------------------------------------------------------------------------------------------------------------'''
class LayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last",
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ASPPPoolingH(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__(
            nn.AdaptiveAvgPool2d((32, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPPoolingW(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__(
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class rot_attconv(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 inp=False,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        self.inp = inp

        self.deconv1 = ConvModule(
            in_channels, n_filters, (1, 9), padding=(0, 4),norm_cfg=None,act_cfg=None)
        self.deconv2 = ConvModule(
            in_channels, n_filters, (9, 1), padding=(4, 0),norm_cfg=None,act_cfg=None)
        self.deconv3 = ConvModule(
            in_channels, n_filters, (9, 1), padding=(4, 0),norm_cfg=None,act_cfg=None)
        self.deconv4 = ConvModule(
            in_channels, n_filters, (1, 9), padding=(0, 4),norm_cfg=None,act_cfg=None)

        self.ASPPH = ASPPPoolingH(in_channels=in_channels, out_channels=n_filters)
        self.ASPPW = ASPPPoolingW(in_channels=in_channels, out_channels=n_filters)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

        self.conv = ConvModule(in_channels * 6,n_filters,1,norm_cfg=None,act_cfg=None)
        self.LN = LayerNorm(n_filters, eps=1e-6, data_format='channels_first',norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.GELU = nn.GELU()


    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x5 = self.ASPPH(x)
        x6 = self.ASPPW(x)
        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)

        if self.inp:
            x = F.interpolate(x, scale_factor=2)

        x = self.conv(x)
        x = self.LN(x)
        x = self.GELU(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

'''------------------------------------------------------------FMCM2----------------------------------------------------------------------'''
class DepthWiseConv2d(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 kernel_size:int=3,
                 padding:int=1,
                 stride:int=1,
                 dilation:int=1,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        # self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
        #                        stride=stride, dilation=dilation, groups=dim_in)
        self.conv1 = ConvModule(dim_in,dim_in, kernel_size=kernel_size,stride=stride,
                                padding=padding,dilation=dilation,groups=dim_in,norm_cfg=None,act_cfg=None)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        # self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.conv2 = ConvModule(dim_in, dim_out, kernel_size=1,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last",
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class MOD_mutil2(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 x:int=8,
                 y:int=8,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 conv_cfg: Optional[dict] = dict(type='Conv1d'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        c_dim_in = dim_in//4
        k_size =3
        pad =(k_size -1) // 2

        self.branch_conv = BRANCH_CONV16(c_dim_in,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.SRU16 = ConvModule(c_dim_in,c_dim_in,1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.params_c = nn.Parameter(torch.Tensor(1, c_dim_in, 1, 1), requires_grad=True)
        nn.init.ones_(self.params_c)
        # self.conv_c = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
        self.conv_c = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1, norm_cfg=None, act_cfg=None)
                                    )

        self.params_x = nn.Parameter(torch.Tensor(1, 1, x, 1), requires_grad=True)
        nn.init.ones_(self.params_x)
        # self.conv_x = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
        self.conv_x = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg,conv_cfg=conv_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1,norm_cfg=None,act_cfg=None,conv_cfg=conv_cfg)
                                    )

        self.params_y = nn.Parameter(torch.Tensor(1, 1, 1, y), requires_grad=True)
        nn.init.ones_(self.params_y)
        # self.conv_y = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
        self.conv_y = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg,conv_cfg=conv_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1,norm_cfg=None,act_cfg=None,conv_cfg=conv_cfg)
                                    )

        self.dw = nn.Sequential(ConvModule(c_dim_in,c_dim_in,1,norm_cfg=None,act_cfg=act_cfg),
                                ConvModule(c_dim_in,c_dim_in,3,1,1,groups=c_dim_in,norm_cfg=None,act_cfg=None)
                                )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first', norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first', norm_cfg=norm_cfg, act_cfg=act_cfg)


        self.ldw = nn.Sequential(ConvModule(dim_in,dim_in,3,1,1,groups=dim_in,norm_cfg=None,act_cfg=act_cfg),
                                 ConvModule(dim_in,dim_out,1,1,norm_cfg=None,act_cfg=None)
             )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        params_c = self.params_c
        x1 = x1 * self.conv_c(F.interpolate(params_c, size=x1.shape[2:4] ,mode='bilinear', align_corners=True))
        x1 = self.branch_conv(x1)

        x2 = x2.permute(0, 3, 1, 2)
        params_x = self.params_x
        x2 = x2 * self.conv_x(F.interpolate(params_x, size=x2.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.branch_conv(x2)

        x3 = x3.permute(0, 2, 1, 3)
        params_y = self.params_y
        x3 = x3 * self.conv_y(F.interpolate(params_y, size=x3.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x3 = self.branch_conv(x3)

        x4 = self.dw(x4)
        x4 = self.branch_conv(x4)

        x = torch.cat([x1,x2,x3,x4],dim=1)

        x = self.norm2(x)
        x = self.ldw(x)

        return x


'''----------------------------------------------------------------------------------------------------------------------'''
class LayerNorm(BaseModule):

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last",
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MOD_coa(BaseModule):
    def __init__(self, in_channels,
                 n_filters,
                 coa_kernel_size: int = 11,
                 inp:bool=False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(in_channels, in_channels // 4, 1, 1, 0, groups=in_channels // 4,
                                bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.inp = inp

        self.deconv1 = ConvModule(
            in_channels // 4, in_channels // 8, (1, coa_kernel_size), padding=(0, coa_kernel_size // 2), norm_cfg=None,
            act_cfg=None)
        self.deconv2 = ConvModule(
            in_channels // 4, in_channels // 8, (coa_kernel_size, 1), padding=(coa_kernel_size // 2, 0), norm_cfg=None,
            act_cfg=None
        )
        self.deconv3 = ConvModule(
            in_channels // 4, in_channels // 8, (coa_kernel_size, 1), padding=(coa_kernel_size // 2, 0), norm_cfg=None,
            act_cfg=None
        )
        self.deconv4 = ConvModule(
            in_channels // 4, in_channels // 8, (1, coa_kernel_size), padding=(0, coa_kernel_size // 2), norm_cfg=None,
            act_cfg=None
        )

        self.conv_ = ConvModule(in_channels // 2, in_channels // 2, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(
            in_channels // 2, n_filters, 1, 1, norm_cfg=None, act_cfg=None)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

        self.LN = LayerNorm(n_filters, eps=1e-6, data_format='channels_first',norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.GELU = build_activation_layer(act_cfg)


    def forward(self, x):
        x = self.conv1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        if self.inp:
            x = F.interpolate(x, scale_factor=2)

        x = self.conv_(x)
        x = self.conv3(x)
        x = self.LN(x)
        x = self.GELU(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


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

"""-----------------------------------------------------final_rot_conv------------------------------------------------"""
class Final_DecoderBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 in_p=True,
                 strip=9,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(Final_DecoderBlock, self).__init__(init_cfg)
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

'''------------------------------------------------------Net-------------------------------------------------------------'''

@MODELS.register_module()
class EFMOD(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=32,
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

        self.conv1 = MFE_Block(self.input_channel,out_channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)  #(bs,c,h,w)→(bs,32,h,w)


        self.conv2 = nn.Sequential(
            # (bs,32,h,w)→(bs,32,h/2,w/2)
            EFE(out_channels[0],out_channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )  # (bs,32,h,w)→(bs,64,h/2,w/2)
        self.conv1x1_1 = ConvModule(out_channels[1]*2,out_channels[1],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv3 = nn.Sequential(
            # (bs,64,h/2,w/2)→(bs,64,h/4,w/4)
            EFE(out_channels[1],out_channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )   # (bs,64,h/4,w/4)→(bs,128,h/4,w/4)
        self.conv1x1_2 = ConvModule(out_channels[2]*2,out_channels[2],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.conv4 = nn.Sequential(
            # (bs,128,h/4,w/4)→(bs,128,h/8,w/8)
            EFE(out_channels[2],out_channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )   # (bs,128,h/4,w/4)→(bs,256,h/8,w/8)
        self.conv1x1_3 = ConvModule(out_channels[3]*2,out_channels[3],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.conv5 = nn.Sequential(
            # (bs,256,h/8,w/8)→(bs,256,h/16,w/16)
            EFE(out_channels[3],out_channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )   # (bs,256,h/8,w/8)→(bs,512,h/16,w/16).
        self.conv1x1_4 = ConvModule(out_channels[4]*2,out_channels[4],1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv5 = ASPP(out_channels[4],[1,6,12,18],out_channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.connect1 = connecter(3,64,0.5,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.connect2 = connecter(3,128,0.25,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.connect3 = connecter(3,256,0.125,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.connect4 = connecter(3,512,0.0625,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.decoder4 = MOD_mutil2(out_channels[3]*2, out_channels[3]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = MOD_mutil2(out_channels[2]*2, out_channels[2]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = MOD_mutil1(out_channels[1]*2, out_channels[1]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = MOD_mutil1(out_channels[0]*2, out_channels[0]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv5_4 = nn.ConvTranspose2d(out_channels[4], out_channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv4_3 = nn.ConvTranspose2d(out_channels[3], out_channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv3_2 = nn.ConvTranspose2d(out_channels[2], out_channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv2_1 = nn.ConvTranspose2d(out_channels[1], out_channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = MOD_coa(out_channels[3]*2, out_channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)   # (bs,512,h,w)→(bs,256,h/8,w/8)

        self.conv7 = MOD_coa(out_channels[2]*2, out_channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg)   # (bs,256,h,w)→(bs,128,h/4,w/4)

        self.conv8 = MOD_coa(out_channels[1]*2, out_channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)   # (bs,128,h,w)→(bs,64,h/2,w/2)

        # self.conv9 = Final_DecoderBlock(out_channels[0]*2, out_channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg) # (bs,64,h/2,w/2)→(bs,32,h,w)
        self.conv9 = nn.Sequential(ConvModule(out_channels[0]*2, out_channels[0],3,1,1,
                                              norm_cfg=None,act_cfg=None),
                                   ConvModule(out_channels[0],out_channels[0],3,1,1,
                                              norm_cfg=norm_cfg,act_cfg=act_cfg))
        self.conv10 = ConvModule(out_channels[0], self.class_num, kernel_size=1, stride=1,norm_cfg=None,act_cfg=None)  # (bs,32,h,w)→(bs,num_classes,h,w)


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

        deconv5 = self.deconv5_4(deconv5)                       # (bs,1024,h/16,w/16)→(bs,1024,h/8,w/8)


        conv6 = torch.cat((deconv5, conv4), 1)      # (bs,1024,h/8,w/8)→(bs,1536,h/8,w/8)

        conv6 = self.decoder4(conv6)
        deconv4 = self.conv6(conv6)                             # (bs,1536,h/8,w/8)→(bs,512,h/8,w/8)
        out.append(conv6)                                       # index=1

        deconv4 = self.deconv4_3(deconv4)                       # (bs,512,h/8,w/8)→(bs,512,h/4,w/4)


        conv7 = torch.cat((deconv4, conv3),1)       # (bs,512,h/8,w/8)→(bs,768,h/4,w/4)
        conv7 = self.decoder3(conv7)
        out.append(conv7)                                       # index=2
        deconv3 = self.conv7(conv7)                             # (bs,768,h/4,w/4)→(bs,256,h/4,w/4)


        deconv3 = self.deconv3_2(deconv3)                       # (bs,256,h/4,w/4)→(bs,256,h/2,w/2)

        conv8 = torch.cat((deconv3, conv2), 1)      # (bs,384,h/2,w/2)→(bs,128,h/2,w/2)
        conv8 = self.decoder2(conv8)
        deconv2 = self.conv8(conv8)                             # (bs,128,h/2,w/2)→(bs,64,h/2,w/2)
        out.append(deconv2)                                       # index=3
        # print(deconv2.shape)


        deconv2 = self.deconv2_1(deconv2)                       # (bs,64,h/2,w/2)→(bs,64,h,w)

        conv9 = torch.cat((deconv2, conv1),  1)     # (bs,64,h,w)→(bs,96,h,w)
        conv9 = self.decoder1(conv9)
        deconv1 = self.conv9(conv9)                             # (bs,64,h,w)→(bs,32,h,w)

        output = self.conv10(deconv1)
        out.append(output)                                       # index=4

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
    moudle = EFMOD()
    print(moudle)
    # moduel = RMFMNet_nor_backon(3).cuda()

    # print(moduel(x)[4].shape)


