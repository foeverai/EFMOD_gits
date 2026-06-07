# Copyright (c) OpenMMLab. All rights reserved.
import math
from mmseg.registry import MODELS
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_wavelets import DWTForward
import einops
from mmengine.model import BaseModule, constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmengine.model.weight_init import trunc_normal_init, normal_init
from mmengine.logging import MMLogger
from einops import rearrange
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


"""---------------------------------------ASPP-----------------------------------------------"""
class ASPP(BaseModule):
    def __init__(self,
                 in_channel:int,
                 rate:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        depth = in_channel
        super().__init__(init_cfg)
        self.rate = rate
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvModule(in_channel, depth, 1, 1,act_cfg=None,norm_cfg=None)
        self.atrous_block1 = ConvModule(in_channel, depth, 1, 1,act_cfg=None,norm_cfg=None)
        self.atrous_block6 = ConvModule(in_channel, depth, 3, 1, padding=self.rate[0], dilation=self.rate[0], act_cfg=None,norm_cfg=None)
        self.atrous_block12 = ConvModule(in_channel, depth, 3, 1, padding=self.rate[1], dilation=self.rate[1],act_cfg=None,norm_cfg=None)
        self.atrous_block18 = ConvModule(in_channel, depth, 3, 1, padding=self.rate[2], dilation=self.rate[2],act_cfg=None,norm_cfg=None)
        self.conv_1x1_output = ConvModule(depth * 5, depth, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)


    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net

"""-----------------------------------SFA---------------------------------------------------"""


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
        return y


class RFCBAMConv(BaseModule):
    def __init__(self,
                 in_channel:int,
                 out_channel:int,
                 kernel_size:int=5,
                 stride:int=1,
                 dilation:int=16,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        if kernel_size % 2 == 0:
            assert ("the kernel_size must be  odd.")
        self.kernel_size = kernel_size
        self.generate = ConvModule(in_channel, in_channel * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                  stride=stride, groups=in_channel, bias=False,norm_cfg=None, act_cfg=None)
        self.dw_conv3_3 = ConvModule(in_channel, in_channel * (kernel_size ** 2), 3, 1, padding=1, groups=in_channel,
                                    bias=False, norm_cfg=None, act_cfg=None)
        self.dw_conv7_7 = ConvModule(in_channel, in_channel * (kernel_size ** 2), 7, 1, padding=3, groups=in_channel,
                                    bias=False,norm_cfg=None,act_cfg=None)
        self.batch_norm_relu = ConvModule(in_channel * (kernel_size ** 2),in_channel * (kernel_size ** 2),1,1,groups=in_channel * (kernel_size ** 2),
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.get_weight = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.se = SE(in_channel, ratio=dilation,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv = ConvModule(in_channel, out_channel, kernel_size, stride=kernel_size, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        b, c = x.shape[0:2]
        channel_attention = self.se(x)
        generate_feature = self.generate(x) + self.dw_conv3_3(x) + self.dw_conv7_7(x)
        generate_feature = self.batch_norm_relu(generate_feature)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        unfold_feature = generate_feature * channel_attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))
        conv_data = unfold_feature * receptive_field_attention
        out = self.conv(conv_data)

        return out

"""--------------------------------------Connect--------------------------------------------------------"""
class Connect_Block(BaseModule):
    def __init__(self, in_channels:int,
                 n_filters:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(in_channels,in_channels//4,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv1 = ConvModule(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4), norm_cfg=None, act_cfg=None
        )
        self.deconv2 = ConvModule(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0), norm_cfg=None, act_cfg=None
        )
        self.deconv3 = ConvModule(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0), norm_cfg=None, act_cfg=None
        )
        self.deconv4 = ConvModule(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4), norm_cfg=None, act_cfg=None
        )
        norm_name, norm = build_norm_layer(norm_cfg, in_channels // 4 + in_channels // 4)
        activate = build_activation_layer(act_cfg)
        self.bn2_relu2 = nn.Sequential(norm, activate)

        self.conv3 = ConvModule(
            in_channels // 4 + in_channels // 4, n_filters, 1,norm_cfg=norm_cfg, act_cfg=act_cfg)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.bn2_relu2(x)
        x = self.conv3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
"""--------------------------------------DWT-----------------------------------------------------------"""
class Down_wt(BaseModule):
    def __init__(self,
                 in_ch:int,
                 out_ch:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg:dict=None
                 ):
        super().__init__(init_cfg)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = ConvModule(in_ch*4, out_ch, kernel_size=1, stride=1,norm_cfg=norm_cfg, act_cfg=act_cfg)
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x
"""----------------------------------------------DSConv----------------------------------------------------"""
class DSConv(BaseModule):
    def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1, act=True,init_cfg=None):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        """
        super(DSConv, self).__init__(init_cfg)
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]?coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """
    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

"""------------------------------------------------------------------------------------------------------------------"""

# backbone

class Encoder(BaseModule):
    def __init__(self,
                 in_chs:int,
                 out_chs:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg:dict=None):
        """
        Building the basic unit of backbone
        :param in_chs:input channels
        :param out_chs:output channels of the module
        """
        super().__init__(init_cfg)
        self.conv = ConvModule(in_chs, out_chs, kernel_size=3, padding=1,norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.gn = nn.GroupNorm(out_chs // 4, out_chs)
        # _,self.gn = build_norm_layer(norm_cfg,out_chs)
        _,self.relu = build_norm_layer(norm_cfg, out_chs)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

# backbone_base_block
class DSMut_Block(BaseModule):

    def __init__(self,
                 in_chs:int,
                 num:int,
                 kernel_size:int,
                 extend:int=1,
                 if_offset:bool=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg:dict=None
                 ):
        """
        The basic unit of the decoder
        :param in_chs:input channels
        :param num:output channels
        :param kernel_size:The convolutional kernel size of DSConv
        :param extend:the range to expand (default 1 for this method)
        :param if_offset:whether deformation is required, if it is False, it is the standard convolution kernel
        """
        super().__init__(init_cfg)
        self.num = num
        # self.out_chs = out_chs
        self.if_offset = if_offset
        self.kernel_size = kernel_size
        self.extend = extend
        # stage
        self.conv0 = Encoder(in_chs, self.num,norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1x = DSConv(in_chs, self.num,0, self.kernel_size,self.if_offset,self.extend)
        self.conv1y = DSConv(in_chs, self.num, 1, self.kernel_size, self.if_offset, self.extend)
        self.conv1_3 = ConvModule(self.num, self.num, 3, padding=1, stride=1, bias=True, groups=self.num, norm_cfg=None, act_cfg=None)
        self.conv1_5 = ConvModule(self.num, self.num, 5, padding=2, stride=1, bias=True, groups=self.num, norm_cfg=None, act_cfg=None)
        self.conv1_7 = ConvModule(self.num, self.num, 7, padding=3, stride=1, bias=True, groups=self.num, norm_cfg=None, act_cfg=None)
        self.conv1_1 = ConvModule(self.num, self.num, 1,stride=1, groups=self.num, norm_cfg=norm_cfg, act_cfg=act_cfg)  # ??????
        self.conv1_out = Encoder(self.num, self.num, norm_cfg=norm_cfg, act_cfg=act_cfg)
    def forward(self, x):
        x_1 = self.conv0(x)
        # x_s1 = self.conv1x(x)
        # x_s2 = self.conv1y(x)
        # x_s3 = x_1+x_s1+x_s2

        x_s4 = self.conv1_3(x_1)
        x_s5 = self.conv1_5(x_1)
        x_s6 = self.conv1_7(x_1)
        x_s7 = self.conv1_1(x_1)

        out1_x = self.conv1_out(x_s4+x_s5+x_s6+x_s7)
        return out1_x    # [batch_size, num, w, h]

class Decoder_Basic(BaseModule):
    def __init__(self,
                 in_chs:int,
                 out_chs:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg:dict=None):
        super(Decoder_Basic, self).__init__(init_cfg)
        self.conv = ConvModule(in_chs, out_chs, 3, padding=1,norm_cfg=None, act_cfg=None)
        self.gn = nn.GroupNorm(out_chs // 4, out_chs)
        # _,self.gn = build_norm_layer(norm_cfg,out_chs)
        _,self.relu = build_norm_layer(norm_cfg,out_chs)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x




class Decoder_Block(BaseModule):
    def __init__(self,
                 in_planes:int,
                 out_planes:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg:dict=None):
        """
        The basic unit of the decoder
        :param in_planes:input channels
        :param out_planes: output channels
        """
        super().__init__(init_cfg)
        self.in_planes = in_planes
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.connect_conv = Connect_Block(int((in_planes/2)*3), out_planes,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.dsconv_x = DSConv(out_planes, out_planes, 0,9,True,1,True)
        self.dsconv_y = DSConv(out_planes, out_planes, 1,9,True,1,True)
        self.conv1 = Decoder_Basic(out_planes*3, out_planes,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x, x_encoder):
        x_1 = self.upsample(x)
        x = torch.cat([x_encoder, x_1], dim=1)  # channel=3/2x.cha
        x = self.connect_conv(x)   # 256
        x_sx = self.dsconv_x(x)    # 256
        x_sy = self.dsconv_y(x)    # 256
        x = self.conv1(torch.cat([x, x_sx, x_sy], dim=1))
        return x



# DSCANet
@MODELS.register_module()
class DSCANet(BaseModule):
    def __init__(self,
                 in_channels:int,
                 base_channels:int,
                 norm_eval: bool = False,  # ?????????????????false
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg: dict = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ],
                 ):
        """
        Building the overall architecture of the network
        :param in_chs:input channels of input img_dir
        :param out_chs:output channels
        """
        super().__init__(init_cfg)
        self.norm_eval = norm_eval
        self.conv0 = ConvModule(in_channels,64,3,1,1,bias=False,norm_cfg=None,act_cfg=None)
        self.conv1 = DSMut_Block(64,64, 9, 1, True, norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.conv1 = ConvModule(64,64,3,1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.attention1 = RFCBAMConv(64,64,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.Down_simple_1 = Down_wt(64,128,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv2 = DSMut_Block(128, 128, 9,1,True,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.conv2 = ConvModule(128,128,3,1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.attention2 = RFCBAMConv(128,128,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.Down_simple_2 = Down_wt(128,256,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv3 = DSMut_Block(256, 256, 9,1,True,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.conv3 = ConvModule(256,256,3,1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.Down_simple_3 = Down_wt(256,512,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.conv4 = DSMut_Block(512, 512, 9,1,True,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.conv4 = ConvModule(512,512,3,1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)


        self.aspp = ASPP(512, (2, 4, 6),norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv2_1 = Encoder(512,512)
        """-------------------------------------------------------------------------------------------"""
        self.decoder_1 = Decoder_Block(512, 256,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder_2 = Decoder_Block(256, 128,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder_3 = Decoder_Block(128, 64,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.final_conv = nn.Sequential(nn.Conv2d(64,base_channels,1,stride=1),
        #                                 nn.SyncBatchNorm(base_channels),
        #                                 nn.GELU()
        #                                 )
        self.final_conv = ConvModule(64,base_channels,1,stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x):
        out = []
        x = self.conv0(x)
        # stage 1
        x_10 = self.conv1(x)
        x_10 = self.attention1(x_10)
        # print(x_10.shape)
        # stage 2
        x_20 = self.Down_simple_1(x_10)  # Reduce size by half and double the number of channels
        # print(x_20.shape)
        x_21 = self.conv2(x_20)
        x_21 = self.attention2(x_21)
        # print(x_21.shape)
        # stage 3
        x_30 = self.Down_simple_2(x_21)  # Reduce size by half and double the number of channels
        # print(x_30.shape)
        x_31 = self.conv3(x_30)
        # print(x_31.shape)
        # stage 4
        x_40 = self.Down_simple_3(x_31)  # Reduce size by half and double the number of channels
        # print(x_40.shape)
        x_41 = self.conv4(x_40)
        # print(x_41.shape)
        # AGR
        x_51 = self.aspp(x_41)
        x_54 = self.conv2_1(x_51)
        # print(x_54.shape)
        # out.append(x_54)
        # decoder
        # step1
        x_61 = self.decoder_1(x_54, x_31)
        # out.append(x_61)
        # print(x_61.shape)
        # step2
        x_62 = self.decoder_2(x_61, x_21)
        out.append(x_62)
        # print(x_62.shape)
        # step3
        x = self.decoder_3(x_62, x_10)
        # print(x.shape)
        # step4
        x = self.final_conv(x)
        # print(x.shape)
        out.append(x)


        return out

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch.')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

# if __name__ == '__main__':
#     x = torch.randn(1, 3, 64, 64).cuda()
#     model = DSCANet(3,8).cuda()
#     print(model(x).shape)



