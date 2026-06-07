import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

class Connect_block(BaseModule):
    def __init__(self, in_channels,
                 n_filters,
                 # BatchNorm,
                 coa_kernel_size: int = 9,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(Connect_block, self).__init__(init_cfg)
        self.conv1 = ConvModule(in_channels, in_channels // 4, 1, 1, 0, groups=in_channels // 4,
                                bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.inp = inp

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
            in_channels // 2, n_filters, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv_(x)
        x = self.conv3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)