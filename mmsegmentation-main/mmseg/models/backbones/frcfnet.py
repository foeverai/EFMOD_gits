import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from .resnet import ResNetV1d

'''------------------------------------------SRU-----------------------------------------------------'''
class GroupBatchnorm2d16(BaseModule):
    def __init__(self,
                 c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10,
                 init_cfg:Optional[dict] = None
                 ):
        super(GroupBatchnorm2d16, self).__init__(init_cfg)
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.contiguous().view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU16(BaseModule):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        self.gn = GroupBatchnorm2d16(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct2(x_1, x_2)

        return x


    def reconstruct2(self, x_1, x_2):

        x_1_part = x_1[:, :3 * x_1.size(1) // 4, :, :]
        x_2_part = x_2[:, :x_2.size(1) // 4, :, :]
        return torch.cat((x_1_part, x_2_part), dim=1)


class GroupBatchnorm2d4(BaseModule):
    def __init__(self,
                 c_num: int,
                 group_num: int = 4,
                 eps: float = 1e-10,
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
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        self.gn = GroupBatchnorm2d4(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct2(x_1, x_2)
        return x


    def reconstruct2(self, x_1, x_2):

        x_1_part = x_1[:, :3 * x_1.size(1) // 4, :, :]
        x_2_part = x_2[:, :x_2.size(1) // 4, :, :]
        return torch.cat((x_1_part, x_2_part), dim=1)


'''-----------------------------------------GMFR1----------------------------------------------------'''
class DepthWiseConv2d(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 kernel_size:int=3,
                 padding:int=1,
                 stride:int=1,
                 dilation:int=1,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
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


class GMFR1(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 x:int=8,
                 y:int=8,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 conv_cfg: Optional[dict] = dict(type='Conv1d'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        c_dim_in = dim_in//4
        k_size =3
        pad =(k_size -1) // 2

        self.SRU4 = SRU4(c_dim_in)

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
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
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


class GMFR2(BaseModule):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 x:int=8,
                 y:int=8,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 conv_cfg: Optional[dict] = dict(type='Conv1d'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        c_dim_in = dim_in//4
        k_size =3
        pad =(k_size -1) // 2

        self.SRU16 = SRU16(c_dim_in)

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


'''----------------------------------------------MCIF------------------------------------------------------------------------'''
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


class MCIF(BaseModule):
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

        self.gamma = nn.Parameter(torch.zeros(1))
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

'''------------------------------------------------------FRCFNet-------------------------------------------------------------'''
class Conv(BaseModule):
    def __init__(self,
                 in_planes,
                 out_planes,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 relu_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(Conv, self).__init__(init_cfg)
        # self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        # self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        # self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.squre = ConvModule(in_planes, out_planes, kernel_size=3, padding=1, stride=1,norm_cfg=None,act_cfg=None)
        self.cross_ver = ConvModule(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1,norm_cfg=None, act_cfg=None)
        self.cross_hor = ConvModule(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1,norm_cfg=None, act_cfg=None)
        # self.bn = nn.BatchNorm2d(out_planes)
        _,self.bn = build_norm_layer(norm_cfg,out_planes)
        # self.ReLU = nn.ReLU(True)
        self.ReLU = build_activation_layer(relu_cfg)


    def forward(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3))

@MODELS.register_module()
class FRCFNet(BaseModule):
    def __init__(self,
                 band_num=3,
                 class_num=1,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='GELU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(FRCFNet, self).__init__(init_cfg)
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'FRCFNet'

        channels = [16, 32, 64, 128, 256]

        self.conv1 = nn.Sequential(
            Conv(self.band_num, channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[0], channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[0], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[1], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[2], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[2], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[2], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[3], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[3], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[3], channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[4], channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg),
            Conv(channels[4], channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.skblock4 = GMFR2(channels[3]*3, channels[3]*3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock3 = GMFR2(channels[2]*3, channels[2]*3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock2 = GMFR1(channels[1]*3, channels[1]*3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock1 = GMFR1(channels[0]*3, channels[0]*3,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv5 = MCIF(256, 256)

        self.deconv5_4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv4_3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv3_2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv2_1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Sequential(Conv(channels[3]*3, channels[3]), Conv(channels[3], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.conv7 = nn.Sequential(Conv(channels[2]*3, channels[2]), Conv(channels[2], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.conv8 = nn.Sequential(Conv(channels[1]*3, channels[1]), Conv(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg))

        self.conv9 = nn.Sequential(Conv(channels[0]*3, channels[0]), Conv(channels[0], channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg))

        # self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)
        self.conv10 = ConvModule(channels[0], self.class_num, kernel_size=1, stride=1,norm_cfg=None,act_cfg=None)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv5 = self.deconv5(conv5)

        deconv5 = self.deconv5_4(deconv5)
        conv5 = self.deconv5_4(conv5)

        conv6 = torch.cat((deconv5, conv4, conv5), 1)
        conv6 = self.skblock4(conv6)
        deconv4 = self.conv6(conv6)

        deconv4 = self.deconv4_3(deconv4)
        conv4 = self.deconv4_3(conv4)

        conv7 = torch.cat((deconv4, conv4, conv3),1)
        conv7 = self.skblock3(conv7)
        deconv3 = self.conv7(conv7)

        deconv3 = self.deconv3_2(deconv3)
        conv3 = self.deconv3_2(conv3)

        conv8 = torch.cat((deconv3, conv3, conv2), 1)
        conv8 = self.skblock2(conv8)
        deconv2 = self.conv8(conv8)

        deconv2 = self.deconv2_1(deconv2)
        conv2 = self.deconv2_1(conv2)

        conv9 = torch.cat((deconv2, conv2, conv1),  1)
        conv9 = self.skblock1(conv9)
        deconv1 = self.conv9(conv9)

        output = F.sigmoid(self.conv10(deconv1))

        return output


if __name__ == '__main__':
    x = torch.randn(2,3,32,32).cuda()
    moduel = FRCFNet(3,1).cuda()
    print(moduel(x).shape)

















