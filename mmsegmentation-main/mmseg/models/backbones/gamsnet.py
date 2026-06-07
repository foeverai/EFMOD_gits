
'''
GAMS_net在实现的时候，
有两个地方SAM 和 CAM 模块
SAM模块： 1*W*H ->C*W*H
CAM模块： C*1*1 ->C*W*H
使用 repeat 或者 title 都会导致梯度叠加
使用参数 replicated： 来选择使用的 repeat or title
        grad_repeat： True ， 需要除以 一个参数 归一化梯度
                      False 使用梯度叠加的 反向传播
为了验证哪一个最好：
GAMSNet_RT: replicated-repeat grad_repeat-True
GAMSNet_TT: replicated-title grad_repeat-True
GAMSNet_RF: replicated-repeat grad_repeat-False
GAMSNet_TF: replicated-title grad_repeat-False
[1] X. Lu, Y. Zhong, Z. Zheng, and L. Zhang, “GAMSNet: Globally aware road detection network with multi-scale residual learning,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 175, pp. 340–352, May 2021, doi: 10.1016/j.isprsjprs.2021.03.008.

'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
from mmseg.registry import MODELS
from pytorch_wavelets import DTCWTForward,DTCWTInverse
from mmcv.cnn.bricks import DropPath
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule
from mmseg.models.utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from functools import partial
from torchvision import models
nonlinearity = partial(F.relu, inplace=True)


class multi_scal_residual_50(BaseModule):
    # 实现子module：Residual Block
    def __init__(self,
                 in_ch:int,
                 out_ch:int,
                 stride:int=1,
                 shortcut:bool=None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(multi_scal_residual_50, self).__init__(init_cfg)
        self.out_ch = out_ch
        self.conv_first = ConvModule(in_ch, out_ch//4, 1, stride, padding=0, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self._1_conv3_3 = ConvModule(out_ch//16, out_ch//16, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self._2_conv3_3 = ConvModule(out_ch//16, out_ch//16, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self._3_conv3_3 = ConvModule(out_ch//16, out_ch//16, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.conv_last = ConvModule(out_ch//4, out_ch, 1, stride, padding=0, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.downsample = shortcut

    def forward(self, x):
        N = x.size()[1]//4
        X = self.conv_first(x)

        x2 = X[:,N:2*N,:,:]
        x3 = X[:,2*N:3*N,:,:]
        x4 = X[:,3*N:4*N,:,:]

        y1 = X[:,:N,:,:]
        y2 = self._1_conv3_3(x2)
        y3 = self._2_conv3_3(x3+y2)
        y4 = self._3_conv3_3(x4+y3)
        y = torch.cat([y1,y2,y3,y4],dim=1)
        out = self.conv_last(y)
        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return out

class Bottleneck(BaseModule):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
        norm_cfg: Optional[dict] = dict(type='BN'),
        act_cfg: Optional[dict] = dict(type='ReLU'),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super(Bottleneck, self).__init__(init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvModule(inplanes, width, kernel_size=1, stride=1, bias=False,norm_cfg=None,act_cfg=None)
        self.bn1 = norm_layer(width)
        ######## multi_scal residual #######################
        self._1_conv3_3 = ConvModule(width//4, width//4, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self._2_conv3_3 = ConvModule(width//4, width//4, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self._3_conv3_3 = ConvModule(width//4, width//4, 3, stride=1, padding=1, bias=False,norm_cfg=norm_cfg,act_cfg=act_cfg)
        ######### end #####################################

        self.conv3 = ConvModule(width, planes * self.expansion, kernel_size=1, stride=stride, bias=False,norm_cfg=None,act_cfg=None)
        _,self.bn3 = build_norm_layer(norm_cfg,planes * self.expansion)
        self.relu = build_activation_layer(act_cfg)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        N = out.size()[1] // 4
        x2 = out[:,N:2*N,:,:]
        x3 = out[:,2*N:3*N,:,:]
        x4 = out[:,3*N:4*N,:,:]

        y1 = out[:,:N,:,:]
        y2 = self._1_conv3_3(x2)
        y3 = self._2_conv3_3(x3+y2)
        y4 = self._3_conv3_3(x4+y3)
        y = torch.cat([y1,y2,y3,y4],dim=1)

        out = self.conv3(y)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GA(BaseModule):
    def __init__(self,
                 in_channel:int,
                 replicated='tile',
                 grad_repeat=False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(GA, self).__init__(init_cfg)

        self.replicated = replicated
        self.grad_repeat = grad_repeat

        self.SAM_first_conv1_1=ConvModule(in_channel, in_channel//16, kernel_size=1, stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.SAM_atrous1_conv3_3 = ConvModule(in_channel // 16, in_channel // 16, kernel_size=3,padding=4, stride=1,dilation=4,
                                              norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.SAM_atrous2_conv3_3 = ConvModule(in_channel // 16, in_channel // 16, kernel_size=3,padding=4, stride=1,dilation=4,
                                              norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.SAM_last_conv1_1 = ConvModule(in_channel//16, 1, kernel_size=1, stride=1,norm_cfg=norm_cfg,act_cfg=act_cfg)


        self.CAM_Global_Average_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        self.CAM_FC1 = nn.Linear(in_channel,in_channel//16)
        self.CAM_FC2 = nn.Linear(in_channel//16,in_channel)


    def forward(self, x):
        _,C,H,W = x.size()
        SA_out = self.SAM_first_conv1_1(x)
        SA_out = self.SAM_atrous1_conv3_3(SA_out)
        SA_out = self.SAM_atrous2_conv3_3(SA_out)
        SA_out = self.SAM_last_conv1_1(SA_out)
        if self.grad_repeat == True:
            grad_decay1 = torch.tensor([1/C],dtype=torch.float).cuda()
        else:
            grad_decay1 = torch.tensor([1],dtype=torch.float).cuda()
        if self.replicated == 'tile':
            SA_out_replicated = SA_out.tile(1,C,1,1) * grad_decay1 #Fs
        else:
            SA_out_replicated = SA_out.repeat(1,C,1,1) * grad_decay1 #Fs
        # [repeat or tile]
        CAM_out = self.CAM_Global_Average_pooling(x)
        CAM_out = CAM_out.squeeze()
        CAM_out = self.CAM_FC1(CAM_out)
        CAM_out = self.CAM_FC2(CAM_out)
        CAM_out1 = CAM_out.unsqueeze(dim=-1).unsqueeze(dim=-1)

        if self.grad_repeat == True:
            # grad_decay2 = torch.tensor([1/(H*W)],dtype=torch.float).cuda()
            grad_decay2 = torch.tensor([1 / (H * W)], dtype=torch.float).to(x.device)
        else:
            # grad_decay2 = torch.tensor([1],dtype=torch.float).cuda()
            grad_decay2 = torch.tensor([1], dtype=torch.float).to(x.device)
        if self.replicated == 'tile':
           CAM_out_replicated = CAM_out1.tile(1, 1, H, W) * grad_decay2 #Fc
        else:
           CAM_out_replicated = CAM_out1.repeat(1, 1, H, W) * grad_decay2  # Fc

        Wg = torch.sigmoid(torch.mul(SA_out_replicated,CAM_out_replicated))
        Og = torch.mul(Wg,x) + x
        return Og


class DecoderBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 n_filters,
                 stride=2,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super(DecoderBlock, self).__init__(init_cfg)
        out_pad = 0 if stride == 1 else 1

        self.conv1 = ConvModule(in_channels, in_channels // 4, 1,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=stride, padding=1,
                                          output_padding=out_pad)
        _,self.norm2 = build_norm_layer(norm_cfg,in_channels//4)
        self.relu2 = nonlinearity

        self.conv3 = ConvModule(in_channels // 4, n_filters, 1,norm_cfg=None,act_cfg=None)
        _,self.norm3 = build_norm_layer(norm_cfg,n_filters)
        self.relu3 = nonlinearity


    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

@MODELS.register_module()
class GAMSNet(BaseModule):
    def __init__(self,
                 block = Bottleneck,
                 in_channels=3,
                 base_channels=32,
                 width_per_group = 64,
                 replicated='tile',
                 grad_repeat=False,
                 norm_layer = None,
                 norm_eval: bool = False,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = [dict(type='Kaiming', layer='Conv2d',
                                                  a=math.sqrt(5),
                                                  distribution='uniform',
                                                  mode='fan_in',
                                                  nonlinearity='leaky_relu'),
                                             dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])]  # 初始化配置字典
                 ):
        super(GAMSNet, self).__init__(init_cfg)
        layers = [3, 4, 6, 3]
        filters = [256, 512, 1024, 2048]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.norm_eval = norm_eval
        self.inplanes = 64
        self.dilation = 1
        self.base_width = width_per_group

        self.conv1 = ConvModule(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.MS_res_block1 = self._make_layer(block,  64, layers[0], stride=2)
        self.MS_res_block2 = self._make_layer(block, 128, layers[1], stride=2)
        self.MS_res_block3 = self._make_layer(block, 256, layers[2], stride=2)
        self.MS_res_block4 = self._make_layer(block, 512, layers[3], stride=2)

        self.GA1 = GA(256,replicated,grad_repeat,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.GA2 = GA(512,replicated,grad_repeat,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.GA3 = GA(1024,replicated,grad_repeat,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.decoder4 = DecoderBlock(filters[3], filters[2],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder3 = DecoderBlock(filters[2], filters[1],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder2 = DecoderBlock(filters[1], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.decoder1 = DecoderBlock(filters[0], filters[0],norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2, padding=1,output_padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = ConvModule(32, 32, 3, padding=1,norm_cfg=None,act_cfg=None)
        self.finalrelu2 = nonlinearity
        self.finaldeconv3 = nn.ConvTranspose2d(32, base_channels, 2, stride=2, padding=0,output_padding=0)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.maxpool(x)

        e1 = self.MS_res_block1(x)
        e11 = self.GA1(e1)
        e2 = self.MS_res_block2(e11)
        e22 = self.GA2(e2)
        e3 = self.MS_res_block3(e22)
        e33 = self.GA3(e3)
        e4 = self.MS_res_block4(e33)
        # Decoder
        d4 = self.decoder4(e4) + e3
        output.append(d4)                   # index=0
        d3 = self.decoder3(d4) + e2
        output.append(d3)                   # index=1
        d2 = self.decoder2(d3) + e1
        output.append(d2)                   # index=2
        d1 = self.decoder1(d2)
        output.append(d1)                   # index=3

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finaldeconv3(out)
        output.append(out)                  # index=4

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
    x = torch.randn(1,3,128,128).cuda()
    module = GAMSNet(base_channels=32).cuda()
    out_list = module(x)
