from idlelib.configdialog import changes

import torch
import torch.nn as nn
import math
from functools import partial
from torch.nn import functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from typing import Optional

from mmseg.models.backbones.resnet import ResNetV1d
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm


'''===================================DConv=========================================='''
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

'''===================================CA=============================================='''
class h_sigmoid(BaseModule):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(BaseModule):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(BaseModule):
    def __init__(self,
                 inp,
                 oup,
                 reduction=32,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(CoordAtt, self).__init__(init_cfg)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = ConvModule(inp, mip, kernel_size=1, stride=1, padding=0,norm_cfg=norm_cfg,act_cfg=None)
        self.act = h_swish()

        self.conv_h = ConvModule(mip, oup, kernel_size=1, stride=1, padding=0,norm_cfg=None,act_cfg=None)
        self.conv_w = ConvModule(mip, oup, kernel_size=1, stride=1, padding=0,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
'''========================================================================================='''
class MCB(BaseModule):
    def __init__(self,
                 in_chan,
                 out_chan,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None):
        super(MCB, self).__init__(init_cfg)
        k_size = [3,5,7,9]
        self.mc1 = ConvModule(in_chan,out_chan,kernel_size=k_size[0],stride=1,padding=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc2 = ConvModule(in_chan,out_chan,kernel_size=k_size[1],stride=1,padding=2,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc3 = ConvModule(in_chan,out_chan,kernel_size=k_size[2],stride=1,padding=3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.mc4 = ConvModule(in_chan,out_chan,kernel_size=k_size[3],stride=1,padding=4,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.incept = ConvModule(in_chan,out_chan,kernel_size=1,stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.con = ConvModule(out_chan*5,out_chan,kernel_size=1,stride=1,padding=0,norm_cfg=None,act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.mc1(x)
        x2 = self.mc2(x)
        x3 = self.mc3(x)
        x4 = self.mc4(x)
        x5 = self.incept(x)
        out = self.con(torch.cat([x1, x2, x3, x4, x5], dim=1))
        return out


'''======================================================================================================================'''

def sp_data(img_tensor):
    # 假设tensor4d是你的4D拼接图像张量，形状为(batch_size, channels, height, width_total)
    batch_size = img_tensor.shape[0]

    # 创建一个列表来存储拆分后的图像
    split_images = []

    for i in range(batch_size):
        # 对每个样本进行拆分
        img1_tensor = img_tensor[i, :, :, :64].unsqueeze(0)
        img2_tensor = img_tensor[i, :, :, 64:].unsqueeze(0)
        img_cat = torch.cat((img1_tensor,img2_tensor),dim=1)
        # 将拆分后的图像作为一个元组添加到列表中（或者你可以选择其他数据结构）
        split_images.append(img_cat)
    cat_img = split_images[0]
    for j in range(1,len(split_images)):
        cat_img = torch.cat((cat_img,split_images[j]),dim=0)

    # 现在split_images包含了batch_size个元组，每个元组包含了两张拆分后的图像
    return cat_img

class Down(BaseModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

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
            # self.conv = nn.Sequential(
            #     ACBlock(in_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg),
            #     ACBlock(out_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg),
            # )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv = DoubleConv(in_channels, out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)
            self.conv = nn.Sequential(
                ACBlock(in_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg),
                ACBlock(out_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg),
            )


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


# @MODELS.register_module()
class FCRNet(BaseModule):
    def __init__(self,
                 in_channels:int=3,
                 base_channels:int=32,
                 norm_eval=False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(FCRNet, self).__init__(init_cfg)
        self.band_num = in_channels
        self.class_num = base_channels
        self.norm_eval = norm_eval

        # channels = [32, 64, 128, 256, 512]
        channels = [32, 64, 128, 256, 512]
        ''' torch.Size([1, 64, 256, 256])
            torch.Size([1, 64, 128, 128])
            torch.Size([1, 128, 64, 64])
            torch.Size([1, 256, 32, 32])
            torch.Size([1, 512, 16, 16])'''
        # self.encoder = ResNetV1d(depth=34)
        self.inc = nn.Sequential(
            ACBlock(in_channels, channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),  # 第一层只有DoubleConv
            ACBlock(channels[0],channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        )       # b,32,512,512
        #
        self.conv12 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[0], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        ) # 64,256,256
        self.conv13 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )   #### 128,128,128
        self.conv14 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )### 128,64,64

        #=====================================
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[0], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[1], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 64,256,256

        self.conv23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 128,128,128
        self.conv24 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )##### 128,64,64

        #=====================================
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 128,128,128

        self.conv34 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )#### 128,64,64


        #====================================
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[3], channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg),
            ACBlock(channels[3], channels[3], norm_cfg=norm_cfg, act_cfg=act_cfg)
        )# 256,64,64
        self.conv45 = ACBlock(channels[3],channels[2],norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg)
        ### 128,64,64


        #=====================================
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[3], channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg),
            ACBlock(channels[4], channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg),
            ACBlock(channels[4], channels[4], norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg),
        )# 512,32,32
        #
        self.conv56 = nn.Sequential(
            nn.ConvTranspose2d(channels[4],channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)
        )### 128,64,64


        self.conv1_1 = ConvModule(channels[2]*5,channels[2],1,norm_cfg=None,act_cfg=None)
        self.ca = CoordAtt(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)     # (b,128,64,64)
        self.mc1 = ACBlock(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg) # 128 64,64
        self.mc2 = MCB(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg) # 128-128,64,64
        #
        self.jum_1 = nn.Sequential(
            nn.ConvTranspose2d(channels[2],channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),
        )# 128,128,128
        self.jum_2 = nn.Sequential(
            nn.ConvTranspose2d(channels[2],channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),
            nn.ConvTranspose2d(channels[2],channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg),
        )# 128,256,256

        self.jum_3 = nn.Sequential(
            nn.ConvTranspose2d(channels[2],channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),
            nn.ConvTranspose2d(channels[2],channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg),
            nn.ConvTranspose2d(channels[2], channels[2], kernel_size=2, stride=2),
            ACBlock(channels[2], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg),
        )# 128,512,512

        self.deconv4 = Up(channels[4]+channels[2],channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)      #512+128->256,s32-64
        self.deconv3 = Up(channels[3]+channels[2],channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)    # 256+128->128,s64-128
        self.deconv2 = Up(channels[2]+channels[2],channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)    # 256+128->128,s128-256
        self.deconv1 = Up(channels[1]+channels[2],channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg)   # 128+128->64,s256-512
        self.deconv0 = nn.Sequential(
                        ACBlock(channels[1],base_channels,norm_cfg=norm_cfg,act_cfg=act_cfg,init_cfg=init_cfg),
                        DoubleConv(base_channels, base_channels, norm_cfg=norm_cfg, act_cfg=act_cfg,init_cfg=init_cfg)
                        )

    def forward(self, x):
        out = []
        e1 = self.inc(x)            # 32,512,512
        e1_jum = self.conv12(e1)
        e1_jum = self.conv13(e1_jum)#128,64,64
        e1_jum = self.conv14(e1_jum)
        # print("e1",e1.shape)
        e2 = self.conv2(e1)
        e2_jum = self.conv23(e2)
        e2_jum = self.conv24(e2_jum)#128,64,64
        # print("e2",e2.shape)        # 64,256,256
        e3 = self.conv3(e2)
        e3_jum = self.conv34(e3)    #128,64,64
        # print("e3",e3.shape)        # 128,128,128
        e4 = self.conv4(e3)
        e4_jum = self.conv45(e4)    # 128,64,64
        # print("e4",e4.shape)        # 256,64,64
        e5 = self.conv5(e4)
        e5_jum = self.conv56(e5)    # 128,64,64
        # print("e5",e5.shape)        # 512,32,32


        ca = self.conv1_1(torch.cat([e1_jum,e2_jum,e3_jum,e4_jum,e5_jum],1)) # 128*5,128
        ca = self.ca(ca)  # 128,64,64
        mc1 = self.mc1(ca)  # 128,64,64
        mc1 = self.mc2(mc1) # 128,64,64

        de1 = self.deconv4(e5,mc1)                    # 256,64,64
        out.append(de1)

        jum1 = self.jum_1(mc1)                         # 128,128,128
        de2 = self.deconv3(de1,jum1)                   # 128,128,128
        out.append(de2)                             # index = 1
        #
        jum2 = self.jum_2(mc1)                       # 128,256,256
        de3 = self.deconv2(de2,jum2)                 # 128,256,256
        out.append(de3)                             # index = 2

        jum3 = self.jum_3(mc1)            # 64,256,256
        de4 = self.deconv1(de3,jum3)                    # 64,512,512
        out.append(de4)                             # index = 3
        #
        de5 = self.deconv0(de4)                     # 32,512,512
        out.append(de5)                             # index = 4

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
    from ptflops import get_model_complexity_info
    x = torch.randn(1,3,512,512).cuda()
    net = FCRNet(in_channels=3,base_channels=32).cuda()
    out = net(x)
    # net = ResNetV1d(depth=34).cuda()

    # summary(net, input_size=(4, 3, 1500, 1500))
    macs, params = get_model_complexity_info(net, (3, 512, 512), print_per_layer_stat=False)
    print(macs, params)

