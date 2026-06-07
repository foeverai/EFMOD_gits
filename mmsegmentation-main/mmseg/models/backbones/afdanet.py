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
from functools import partial
from torchvision import models
from einops import rearrange


from projects.Adabins.configs.adabins.adabins_efficient_b5_4x16_25e_kitti_352x704 import norm_cfg


class ECA(BaseModule):
    def __init__(self,
                 channel,
                 k_size=3,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(ECA, self).__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# Strip Grouped Aggregation Decoder Module
class SGADBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 kernel_size,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(SGADBlock, self).__init__(init_cfg)
        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.bn1 = nn.BatchNorm2d(in_channels // 4)
        # self.relu1 = nonlinearity
        self.conv1 = ConvModule(in_channels,in_channels//4,1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.strip_conv1 = ConvModule(in_channels // 4, in_channels // 8, (1, kernel_size), padding=(0, kernel_size//2),norm_cfg=norm_cfg,act_cfg=None)
        self.strip_conv2 = ConvModule(in_channels // 4, in_channels // 8, (kernel_size, 1), padding=(kernel_size//2, 0),norm_cfg=norm_cfg,act_cfg=None)
        self.strip_conv3 = ConvModule(in_channels // 4, in_channels // 8, (kernel_size, 1), padding=(kernel_size//2, 0),norm_cfg=None,act_cfg=None)
        self.strip_conv4 = ConvModule(in_channels // 4, in_channels // 8, (1, kernel_size), padding=(0, kernel_size//2),norm_cfg=None,act_cfg=None)

        _,self.s3bn = build_norm_layer(norm_cfg,in_channels // 8)
        _,self.s4bn = build_norm_layer(norm_cfg,in_channels // 8)
        self.relu12 = build_activation_layer(act_cfg)
        self.relu34 = build_activation_layer(act_cfg)
        self.eca1 = ECA(in_channels // 4,norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.eca2 = ECA(in_channels // 4,norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)

        self.conv3 = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 2, 3, stride=2, padding=1, output_padding=1
        )
        _,self.bn3 = build_norm_layer(norm_cfg,in_channels // 2)
        self.relu3 = build_activation_layer(act_cfg)

        self.conv4 = nn.Conv2d(in_channels // 2, n_filters, 1)
        _,self.bn4 = build_norm_layer(norm_cfg,n_filters)
        self.relu4 = build_activation_layer(act_cfg)
    def forward(self, x):
        x = self.conv1(x)

        x1 = self.strip_conv1(x)
        x2 = self.strip_conv2(x)
        x12 = self.eca1(self.relu12(torch.cat((x1, x2), 1))) + x

        x3 = self.s3bn(self.inv_h_transform(self.strip_conv3(self.h_transform(x))))
        x4 = self.s4bn(self.inv_v_transform(self.strip_conv4(self.v_transform(x))))
        x34 = self.eca2(self.relu34(torch.cat((x3, x4), 1))) + x
        x = torch.cat((x12, x34), 1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        return x

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

class SGADecoder(BaseModule):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_sizes=[3, 5, 7],
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(SGADecoder, self).__init__(init_cfg)
        self.sgad1 = SGADBlock(in_channels, filters // 2, kernel_sizes[0],norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.sgad2 = SGADBlock(in_channels, filters // 2, kernel_sizes[1],norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.sgad3 = SGADBlock(in_channels, filters // 2, kernel_sizes[2],norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.conv1 = nn.Sequential(
            ConvModule((filters // 2) * 3, filters, 1, bias=False),
            nn.BatchNorm2d(filters)
            )
        self.relu1 = build_activation_layer(act_cfg)
    def forward(self, x):
        x1 = self.sgad1(x)
        x2 = self.sgad2(x)
        x3 = self.sgad3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        return x

# Spatial Aggregation Downsample Block
class Focus(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(Focus, self).__init__(init_cfg)
        self.cfusion = ConvModule(in_channels * 4, out_channels, kernel_size=1, stride=1,norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        x = torch.cat([ x[:, :, 0::2, 0::2],
                        x[:, :, 1::2, 0::2],
                        x[:, :, 0::2, 1::2],
                        x[:, :, 1::2, 1::2]], dim=1)  # x = [H/2, W/2]
        x = self.cfusion(x)     # x = [H/2, W/2]
        return x

class SADBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(SADBlock, self).__init__(init_cfg)
        self.Focus_downsample = Focus(in_channels=in_channels, out_channels=out_channels,norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.dwconv = ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,norm_cfg=None,act_cfg=None)
        self.dwconv_downsample = ConvModule(out_channels, out_channels, kernel_size=7, stride=2, padding=3, groups=out_channels)
        self.dwconv_act = nn.GELU()
        _,self.dwconv_bn = build_norm_layer(norm_cfg,out_channels)
        self.maxpool_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        _,self.maxpool_bn = build_norm_layer(norm_cfg,out_channels)
        self.Aggregation = ConvModule(3 * out_channels, out_channels, kernel_size=1, stride=1,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        c = x
        x = self.dwconv(x)# x = [H, W] --> [H, W]
        m = x

        c = self.Focus_downsample(c)# c = [H, W] --> [H/2, W/2]

        x = self.dwconv_downsample(x)
        x = self.dwconv_act(x)
        x = self.dwconv_bn(x)# x = [H, W] --> [H/2, W/2]

        m = self.maxpool_downsample(m)
        m = self.maxpool_bn(m)# m = [H/2, W/2]

        x = torch.cat([c, x, m], dim=1)# x = [H/2, W/2]
        x = self.Aggregation(x)# x = [H/2, W/2] --> [H/2, W/2]

        return x# x = [H/2, W/2]

# Parallel Channel-Spatial Attention
class PCSA(BaseModule):
    def __init__(self,
                 dim,head_num,
                 patch_size= 8,
                 kernel_sizes= [3, 5, 7, 9],
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(PCSA, self).__init__(init_cfg)
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.kernel_sizes = kernel_sizes
        self.group_channels = self.dim // 4
        group_channels = self.dim // 4
        self.dwc1 = nn.Conv1d(group_channels, group_channels, kernel_size=kernel_sizes[0],
                                   padding=kernel_sizes[0] // 2, groups=group_channels)
        self.dwc2 = nn.Conv1d(group_channels, group_channels, kernel_size=kernel_sizes[1],
                                      padding=kernel_sizes[1] // 2, groups=group_channels)
        self.dwc3 = nn.Conv1d(group_channels, group_channels, kernel_size=kernel_sizes[2],
                                      padding=kernel_sizes[2] // 2, groups=group_channels)
        self.dwc4 = nn.Conv1d(group_channels, group_channels, kernel_size=kernel_sizes[3],
                                      padding=kernel_sizes[3] // 2, groups=group_channels)
        self.sa_gate = nn.Sigmoid()
        self.gn_h = nn.GroupNorm(4, dim)
        self.gn_w = nn.GroupNorm(4, dim)
        self.identity = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1, bias=False, groups=dim,norm_cfg=None,act_cfg=None)
        self.k = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1, bias=False, groups=dim,norm_cfg=None,act_cfg=None)
        self.v = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1, bias=False, groups=dim,norm_cfg=None,act_cfg=None)
        self.sig = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=(patch_size, patch_size), stride=patch_size)

    def forward(self, x):
        # Directional Spatial Attention
        b, c, h_, w_ = x.size()
        x2 = x.clone()
        x_h = x.mean(dim=3)
        feat1_h, feat2_h, feat3_h, feat4_h = torch.split(x_h, self.group_channels, dim=1)
        x_w = x.mean(dim=2)
        feat1_w, feat2_w, feat3_w, feat4_w = torch.split(x_w, self.group_channels, dim=1)

        x_h_att = self.sa_gate(self.gn_h(torch.cat((
            self.dwc1(feat1_h),
            self.dwc2(feat2_h),
            self.dwc3(feat3_h),
            self.dwc4(feat4_h),
        ), dim=1)))
        x_h_att = x_h_att.view(b, c, h_, 1)

        x_w_att = self.sa_gate(self.gn_w(torch.cat((
            self.dwc1(feat1_w),
            self.dwc2(feat2_w),
            self.dwc3(feat3_w),
            self.dwc4(feat4_w)
        ), dim=1)))
        x_w_att = x_w_att.view(b, c, 1, w_)
        x = x * x_h_att * x_w_att

        # Multi-Head Contextual Channel Attention
        y = self.avgpool(x2)
        y = self.identity(y)
        _, _, h_, w_ = y.size()

        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        c_att = q @ k.transpose(-2, -1) * self.scaler
        c_att = c_att.softmax(dim=-1)
        c_att = c_att @ v
        c_att = rearrange(c_att, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        c_att = c_att.mean((2, 3), keepdim=True)
        c_att = self.sig(c_att)
        return x + c_att * x2

#  MLP
class MS_DWConv(BaseModule):
    def __init__(self,
                 dim,
                 scale=(1, 3, 5, 7),
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(MS_DWConv, self).__init__(init_cfg)
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = ConvModule(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels,norm_cfg=None,act_cfg=None)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class MLP(BaseModule):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(MLP, self).__init__(init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            ConvModule(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MS_DWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc2(x)
        return x

# Spatial-Enhanced Adaptive Road Feature Fusion Module
class SEAFModule(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mlp_ratio=4,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(SEAFModule, self).__init__(init_cfg)
        self.sadb = SADBlock(in_channels, out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.p1 = nn.Parameter(torch.tensor(1.0))
        self.p2 = nn.Parameter(torch.tensor(1.0))
        self.psca = PCSA(out_channels,head_num=8,patch_size=8,norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.mlp = MLP(in_features=out_channels, hidden_features=int(out_channels * mlp_ratio),norm_cfg=norm_cfg,
                       act_cfg=act_cfg,init_cfg=init_cfg)
    def forward(self, x1,x2):
        probs = torch.softmax(torch.stack([self.p1, self.p2]), dim=0)
        p1, p2 = probs[0], probs[1]
        x1 = self.act(self.norm1(self.sadb(x1)))
        x1 = p1*x1
        x2 = p2*x2
        x = x1 + x2
        shortcut = x.clone()
        x = self.psca(x) + shortcut
        x = self.mlp(self.norm2(x)) + x
        return x

# main AFDANet
# @MODELS.register_module()
class AFDANet(BaseModule):
    def __init__(self,
                 in_channels=3,
                 img_size=1024,
                 num_classes=1,
                 norm_eval: bool = False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(AFDANet, self).__init__(init_cfg)
        self.norm_eval = norm_eval
        self.img_size=img_size
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.seaf1 = SEAFModule(filters[0],filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.seaf2 = SEAFModule(filters[0],filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.seaf3 = SEAFModule(filters[1],filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.seaf4 = SEAFModule(filters[2],filters[3],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.decoder4 = SGADecoder(filters[3], filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.decoder3 = SGADecoder(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.decoder2 = SGADecoder(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.decoder1 = SGADecoder(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = build_activation_layer(act_cfg)
        # self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
        # self.finalrelu2 = build_activation_layer(act_cfg)
        # self.finalconv3 = ConvModule(32, num_classes, 3, padding=1,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        output = []
        if self.img_size != 1024:
            x = F.interpolate(x, size=1024, mode='bilinear', align_corners=True)
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        encoder1 = self.encoder1(x1)
        f1 = self.seaf1(x,encoder1)
        encoder2 = self.encoder2(encoder1)
        f2 = self.seaf2(f1,encoder2)
        encoder3 = self.encoder3(encoder2)
        f3 = self.seaf3(f2,encoder3)
        encoder4 = self.encoder4(encoder3)
        f4 = self.seaf4(f3,encoder4)
        f4 = encoder4 + f4
        output.append(f4)                   # index = 0
        # Decoder
        d4 = self.decoder4(f4)
        d4 = d4 + f3
        output.append(d4)                   # index = 1
        d3 = self.decoder3(d4)
        d3 = d3 + f2
        output.append(d3)                   # index = 2
        d2 = self.decoder2(d3)
        d2 = d2 + f1
        output.append(d2)                   # index = 3
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        if self.img_size != 1024:
            out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=True)

        output.append(out)                  # index = 4
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

if __name__ == '__main__':
    test_input = torch.randn(1, 3, 512, 512).cuda()
    net = AFDANet(img_size=512).cuda()

    out = net(test_input)
    print(out[4].shape)
