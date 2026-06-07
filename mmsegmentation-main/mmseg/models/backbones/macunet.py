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


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        ConvModule(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,norm_cfg=None,act_cfg=None),
        # nn.ReLU()  # inplace=True
        nn.LeakyReLU()
    )


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

@MODELS.register_module()
class MACUNet(BaseModule):
    def __init__(self,
                 in_channels:int=3,
                 base_channels:int=16,
                 norm_eval=False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super(MACUNet, self).__init__(init_cfg)
        self.band_num = in_channels
        self.class_num = base_channels
        self.norm_eval = norm_eval
        self.name = 'MACUNet'

        # channels = [32, 64, 128, 256, 512]
        channels = [16, 32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            ACBlock(self.band_num, channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[0], channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv12 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[0], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv13 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )
        self.conv14 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[0], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv24 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[1], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[2], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[2], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )
        self.conv34 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[2], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[3], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[3], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ACBlock(channels[3], channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[4], channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[4], channels[4],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.skblock4 = ChannelAttention(channels[3]*5, channels[3]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock3 = ChannelAttention(channels[2]*5, channels[2]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock2 = ChannelAttention(channels[1]*5, channels[1]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.skblock1 = ChannelAttention(channels[0]*5, channels[0]*2, 16,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv43 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv42 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv41 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Sequential(
            ACBlock(channels[4], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[3], channels[3],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv32 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv31 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            ACBlock(channels[3], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[2], channels[2],norm_cfg=norm_cfg,act_cfg=act_cfg),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv21 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            ACBlock(channels[2], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[1], channels[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            ACBlock(channels[1], channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg),
            ACBlock(channels[0], channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        )

        self.conv10 = ConvModule(channels[0], self.class_num, kernel_size=1, stride=1,norm_cfg=None,act_cfg=None)

    def forward(self, x):
        # x = sp_data(x)   # 将拼接的数据进行拆分
        # print(x.shape)
        out = []
        conv1 = self.conv1(x)
        conv12 = self.conv12(conv1)
        conv13 = self.conv13(conv12)
        conv14 = self.conv14(conv13)

        conv2 = self.conv2(conv1)
        conv23 = self.conv23(conv2)
        conv24 = self.conv24(conv23)

        conv3 = self.conv3(conv2)
        conv34 = self.conv34(conv3)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        out.append(conv5)                                       # index=0

        deconv4 = self.deconv4(conv5)
        deconv43 = self.deconv43(deconv4)
        deconv42 = self.deconv42(deconv43)
        deconv41 = self.deconv41(deconv42)

        conv6 = torch.cat((deconv4, conv4, conv34, conv24, conv14), 1)
        conv6 = self.skblock4(conv6)
        conv6 = self.conv6(conv6)
        out.append(conv6)                                       # index=1
        del deconv4, conv4, conv34, conv24, conv14, conv5

        deconv3 = self.deconv3(conv6)
        deconv32 = self.deconv32(deconv3)
        deconv31 = self.deconv31(deconv32)

        conv7 = torch.cat((deconv3, deconv43, conv3, conv23, conv13), 1)
        conv7 = self.skblock3(conv7)
        conv7 = self.conv7(conv7)
        out.append(conv7)                                       # index=2
        del deconv3, deconv43, conv3, conv23, conv13, conv6

        deconv2 = self.deconv2(conv7)
        deconv21 = self.deconv21(deconv2)

        conv8 = torch.cat((deconv2, deconv42, deconv32, conv2, conv12), 1)
        conv8 = self.skblock2(conv8)
        conv8 = self.conv8(conv8)
        out.append(conv8)                                       # index=3
        # print(conv8.shape)
        del deconv2, deconv42, deconv32, conv2, conv12, conv7

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, deconv41, deconv31, deconv21, conv1), 1)
        conv9 = self.skblock1(conv9)
        conv9 = self.conv9(conv9)
        # conv9 = self.seblock(conv9)
        del deconv1, deconv41, deconv31, deconv21, conv1, conv8

        output = self.conv10(conv9)
        out.append(output)                                      # index=4

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
    x = torch.randn(2,3,512,512).cuda()
    net = MACUNet(in_channels=3,base_channels=32).cuda()
    # out = net(x)
    macs, params = get_model_complexity_info(net, (3, 512, 512), print_per_layer_stat=False)
    print(macs, params)
    # print(out[4].shape)
