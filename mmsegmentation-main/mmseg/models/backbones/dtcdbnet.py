import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
from mmseg.registry import MODELS
from mmcv.cnn.bricks import DropPath
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmseg.models.utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
'''-----------------------------------------全局门控单元----------------------------------------------------'''
class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))

"""--------------------------------------------EncoderBlock---------------------------------------------------------"""
class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 9,
            v_kernel_size: int = 9,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x

class DownSamplingLayer(BaseModule):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.down_conv(x)

class InceptionBottleneck(BaseModule):  # 实现论文中的PKI Module
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9),
            dilations: Sequence[int] = (1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        # self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
        #                            autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
        #                            groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x

class Encoder_Block(BaseModule):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9),
            dilations: Sequence[int] = (1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 9,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        if norm_cfg is not None:
            self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]   # 根据norm_cfg创建归一化层，in_channels是归一化层的输入和输出的通道数
            self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]   # 此函数返回的是name,layer(即归一化层)，我们只需要第二个元素及归一化层
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
                           norm_cfg=None, act_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class Enocder_Stage(BaseModule):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9),
            dilations: Sequence[int] = (1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='GELU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels, norm_cfg, act_cfg)

        self.conv1 = ConvModule(out_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True, norm_cfg=None, act_cfg=None) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            Encoder_Block(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity, norm_cfg, act_cfg) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)

        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)

        return z

"""-------------------------------------------------ASPP_aug--------------------------------------------------------"""
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



'''------------------------------------------SRU-----------------------------------------------------'''
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


class SRU16(BaseModule):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        # self.gn = GroupBatchnorm2d16(oup_channels, group_num=group_num,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.gn = nn.GroupNorm(num_groups=group_num,num_channels=oup_channels)
        self.cha_at = SE(oup_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.gate_treshold = gate_treshold
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x= self.gn(x)
        out = self.cha_at(gn_x)
        return out


    def reconstruct2(self, x_1, x_2):

        x_1_part = x_1[:, :3 * x_1.size(1) // 4, :, :]
        x_2_part = x_2[:, :x_2.size(1) // 4, :, :]
        return torch.cat((x_1_part, x_2_part), dim=1)


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


class SRU4(BaseModule):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        # self.gn = GroupBatchnorm2d4(oup_channels, group_num=group_num,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.gn = nn.GroupNorm(group_num,oup_channels)
        self.chan_at = SE(oup_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.gate_treshold = gate_treshold
        # self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        out = self.chan_at(gn_x)
        # w_gamma = self.gn.gamma / sum(self.gn.gamma)
        # reweigts = self.sigomid(gn_x * w_gamma)
        # info_mask = reweigts >= self.gate_treshold
        # noninfo_mask = reweigts < self.gate_treshold
        # x_1 = info_mask * x
        # x_2 = noninfo_mask * x
        # x = self.reconstruct2(x_1, x_2)
        return out


    # def reconstruct2(self, x_1, x_2):
    #
    #     x_1_part = x_1[:, :3 * x_1.size(1) // 4, :, :]
    #     x_2_part = x_2[:, :x_2.size(1) // 4, :, :]
    #     return torch.cat((x_1_part, x_2_part), dim=1)


'''-----------------------------------------GMFR1----------------------------------------------------'''
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


class GMFR1(BaseModule):
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

        self.SRU4 = SRU4(c_dim_in,norm_cfg=norm_cfg,act_cfg=act_cfg)

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
        # self.conv_y = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
        self.conv_y = nn.Sequential(ConvModule(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in,norm_cfg=None,act_cfg=act_cfg,conv_cfg=conv_cfg),
                                    ConvModule(c_dim_in, c_dim_in, 1,norm_cfg=None,act_cfg=None,conv_cfg=conv_cfg)
                                    )

        # self.dw = nn.Sequential(
        #     nn.Conv2d(c_dim_in, c_dim_in, 1),
        #     nn.GELU(),
        #     nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in))
        self.dw = nn.Sequential(ConvModule(c_dim_in,c_dim_in,1,norm_cfg=None,act_cfg=act_cfg),
                                ConvModule(c_dim_in,c_dim_in,3,1,1,groups=c_dim_in,norm_cfg=None,act_cfg=None)

        )
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first',norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first',norm_cfg=norm_cfg,act_cfg=act_cfg)

        # self.ldw = nn.Sequential(
        #     nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
        #     nn.GELU(),
        #     nn.Conv2d(dim_in, dim_out, 1))
        self.ldw = nn.Sequential(ConvModule(dim_in,dim_in,3,1,1,groups=dim_in,norm_cfg=None,act_cfg=act_cfg),
                                 ConvModule(dim_in,dim_out,1,1,norm_cfg=None,act_cfg=None)
             )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        params_c = self.params_c
        x1 = x1 * self.conv_c(F.interpolate(params_c, size=x1.shape[2:4] ,mode='bilinear', align_corners=True))
        x1 = self.SRU4(x1)

        x2 = x2.permute(0, 3, 1, 2)
        params_x = self.params_x
        x2 = x2 * self.conv_x(F.interpolate(params_x, size=x2.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.SRU4(x2)

        x3 = x3.permute(0, 2, 1, 3)
        params_y = self.params_y
        x3 = x3 * self.conv_y(F.interpolate(params_y, size=x3.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x3 = self.SRU4(x3)

        x4 = self.dw(x4)
        x4 = self.SRU4(x4)

        x = torch.cat([x1,x2,x3,x4],dim=1)

        x = self.norm2(x)
        x = self.ldw(x)

        return x

'''------------------------------------------------------------GMFR2----------------------------------------------------------------------'''
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



class GMFR2(BaseModule):
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

        self.SRU16 = SRU16(c_dim_in,norm_cfg=norm_cfg,act_cfg=act_cfg)
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

        # self.dw = nn.Sequential(
        #     nn.Conv2d(c_dim_in, c_dim_in, 1),
        #     nn.GELU(),
        #     nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in))
        self.dw = nn.Sequential(ConvModule(c_dim_in,c_dim_in,1,norm_cfg=None,act_cfg=act_cfg),
                                ConvModule(c_dim_in,c_dim_in,3,1,1,groups=c_dim_in,norm_cfg=None,act_cfg=None)
                                )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first', norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first', norm_cfg=norm_cfg, act_cfg=act_cfg)

        # self.ldw = nn.Sequential(
        #     nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
        #     nn.GELU(),
        #     nn.Conv2d(dim_in, dim_out, 1))
        self.ldw = nn.Sequential(ConvModule(dim_in,dim_in,3,1,1,groups=dim_in,norm_cfg=None,act_cfg=act_cfg),
                                 ConvModule(dim_in,dim_out,1,1,norm_cfg=None,act_cfg=None)
             )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        params_c = self.params_c
        x1 = x1 * self.conv_c(F.interpolate(params_c, size=x1.shape[2:4] ,mode='bilinear', align_corners=True))
        x1 = self.SRU16(x1)

        x2 = x2.permute(0, 3, 1, 2)
        params_x = self.params_x
        x2 = x2 * self.conv_x(F.interpolate(params_x, size=x2.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.SRU16(x2)

        x3 = x3.permute(0, 2, 1, 3)
        params_y = self.params_y
        x3 = x3 * self.conv_y(F.interpolate(params_y, size=x3.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x3 = self.SRU16(x3)

        x4 = self.dw(x4)
        x4 = self.SRU16(x4)

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


# class ASPPPoolingH(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPoolingH, self).__init__(
#             nn.AdaptiveAvgPool2d((32, 1)),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU())
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# class ASPPPoolingW(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPoolingW, self).__init__(
#             nn.AdaptiveAvgPool2d((1, 32)),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU())
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Conv(BaseModule):
    def __init__(self, in_channels,
                 n_filters,
                 coa_kernel_size: int = 11,
                 inp:bool=False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(Conv, self).__init__(init_cfg)
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

'''------------------------------------------------------Net-------------------------------------------------------------'''
class Stem_block(BaseModule):
    def __init__(self,
                 in_ch:int,
                 out_ch:int,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 relu_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(in_ch, out_ch, kernel_size=3, padding=1, stride=1, norm_cfg=None, act_cfg=None)
        self.cross_ver = ConvModule(in_ch, out_ch, kernel_size=(1, 3), padding=(0, 1), stride=1, norm_cfg=None,
                                    act_cfg=None)
        self.cross_hor = ConvModule(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0), stride=1, norm_cfg=None,
                                    act_cfg=None)
        _, self.bn = build_norm_layer(norm_cfg, out_ch)
        self.ReLU = build_activation_layer(relu_cfg)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3))

@MODELS.register_module()
class DTCBDNet(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
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
        super(DTCBDNet, self).__init__(init_cfg)
        self.input_channel = in_channels
        self.class_num = base_channels
        self.norm_eval = norm_eval

        out_channels = [32, 64, 128, 256, 512]

        self.conv1 = Stem_block(self.input_channel,out_channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)  #(bs,c,h,w)→(bs,32,h,w)

        self.conv2 = Enocder_Stage(out_channels[0],out_channels[1],num_blocks=4,norm_cfg=norm_cfg,act_cfg=act_cfg)  #(bs,c,h,w)→(bs,64,h/2,w/2)

        self.conv3 = Enocder_Stage(out_channels[1],out_channels[2],12,norm_cfg=norm_cfg,act_cfg=act_cfg)   #(bs,64,h/2,w/2)→(bs,128,h/4,w/4)

        self.conv4 = Enocder_Stage(out_channels[2],out_channels[3],20,norm_cfg=norm_cfg,act_cfg=act_cfg)    #(bs,128,h/4,w/4)→(bs,256,h/8,w/8)

        self.conv5 = Enocder_Stage(out_channels[3],out_channels[4],4,norm_cfg=norm_cfg,act_cfg=act_cfg)   #(bs,256,h/8,w/8)→(bs,512,h/16,w/16)

        self.deconv5 = ASPP(out_channels[4],[6,12,18],out_channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.decoder4 = GMFR2(out_channels[3]*2, out_channels[3]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = GMFR2(out_channels[2]*2, out_channels[2]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = GMFR1(out_channels[1]*2, out_channels[1]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = GMFR1(out_channels[0]*2, out_channels[0]*2,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv5_4 = nn.ConvTranspose2d(out_channels[4], out_channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv4_3 = nn.ConvTranspose2d(out_channels[3], out_channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv3_2 = nn.ConvTranspose2d(out_channels[2], out_channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv2_1 = nn.ConvTranspose2d(out_channels[1], out_channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = Conv(out_channels[3]*2, out_channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)                                                   #(bs,512,h,w)→(bs,256,h/8,w/8)

        self.conv7 = Conv(out_channels[2]*2, out_channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg)                                                   #(bs,256,h,w)→(bs,128,h/4,w/4)

        self.conv8 = Conv(out_channels[1]*2, out_channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)                                                   #(bs,128,h,w)→(bs,64,h/2,w/2)

        self.conv9 = Conv(out_channels[0]*2, out_channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)                                                   #(bs,64,h/2,w/2)→(bs,32,h,w)
        # self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)
        self.conv10 = ConvModule(out_channels[0], self.class_num, kernel_size=1, stride=1,norm_cfg=None,act_cfg=None)  #(bs,32,h,w)→(bs,num_classes,h,w)


    def forward(self, x):
        x = x.contiguous()
        out = []
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv5 = self.deconv5(conv5)
        out.append(deconv5)                                     # index=0

        deconv5 = self.deconv5_4(deconv5)                       # (bs,512,h/16,w/16)→(bs,512,h/8,w/8)
        # conv5 = self.deconv5_4(conv5)

        conv6 = torch.cat((deconv5, conv4), 1)      # (bs,512,h/8,w/8)→(bs,768,h/8,w/8)
        conv6 = self.decoder4(conv6)
        deconv4 = self.conv6(conv6)                             # (bs,768,h/8,w/8)→(bs,256,h/8,w/8)
        out.append(deconv4)                                     # index=1

        deconv4 = self.deconv4_3(deconv4)                       # (bs,256,h/8,w/8)→(bs,256,h/4,w/4)


        conv7 = torch.cat((deconv4, conv3),1)       # (bs,256,h/8,w/8)→(bs,384,h/4,w/4)
        conv7 = self.decoder3(conv7)
        deconv3 = self.conv7(conv7)                             # (bs,384,h/4,w/4)→(bs,128,h/4,w/4)
        out.append(deconv3)                                     # index=2

        deconv3 = self.deconv3_2(deconv3)                       # (bs,128,h/4,w/4)→(bs,128,h/2,w/2)


        conv8 = torch.cat((deconv3, conv2), 1)      # (bs,128,h/2,w/2)→(bs,192,h/2,w/2)
        conv8 = self.decoder2(conv8)
        deconv2 = self.conv8(conv8)                             # (bs,192,h/2,w/2)→(bs,64,h/2,w/2)
        out.append(deconv2)                                     # index=3

        deconv2 = self.deconv2_1(deconv2)                       # (bs,64,h/2,w/2)→(bs,64,h,w)


        conv9 = torch.cat((deconv2, conv1),  1)     # (bs,64,h,w)→(bs,96,h,w)
        conv9 = self.decoder1(conv9)
        deconv1 = self.conv9(conv9)                             # (bs,96,h,w)→(bs,32,h,w)

        # output = F.sigmoid(self.conv10(deconv1))
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
    moduel = DTCBDNet(3,32).cuda()
    print(moduel(x).shape)

















