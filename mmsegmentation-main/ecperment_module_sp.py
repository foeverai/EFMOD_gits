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


'''-----------------------------------------------------------------------------------------------------------------'''
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


"""-----------------------------------------------------------------------------------------------------------------"""
class stem_block(BaseModule):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 dilation=1,
                 bias = False,
                 # strip=9,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None
                 ):
        super().__init__(init_cfg)

        self.conv_a0 = DoubleConv(in_ch,out_ch,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.multi_conv1 = ConvModule(out_ch, out_ch//4, 3, stride=stride,padding=1,norm_cfg=None,act_cfg=None)
        self.multi_conv2 = ConvModule(out_ch, out_ch//4, 3, stride=stride,padding=1,norm_cfg=None,act_cfg=None)
        self.multi_conv3 = ConvModule(out_ch, out_ch//4, 3, stride=stride,padding=1,norm_cfg=None,act_cfg=None)
        self.multi_conv4 = ConvModule(out_ch, out_ch//4, 3, stride=stride,padding=1,norm_cfg=None,act_cfg=None)

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
        out = self.channel_concern(out)
        return out+x

'''------------------------------------------------------------------------------------------------------------------'''
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
'''----------------------------------------------------------------------------------------------------------------------'''
class SPConv_Block(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 down=False,
                 sp_num: int = 4,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 ):
        super().__init__(init_cfg)
        self.sp_num = sp_num
        self.padding_mode = 'constant'                              # 使用常数0进行填充
        self.padding_value = 0
        if down:
            self.down_sample = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          norm_cfg=norm_cfg,act_cfg=act_cfg)
        else:
            self.down_sample = None
        self.block_conv = DoubleConv(out_channels,out_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.cat_conv = nn.Sequential(
            ConvModule(out_channels*2, out_channels,kernel_size=3, padding=1,bias=False,norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, bias=False, norm_cfg=norm_cfg,
                       act_cfg=None),
        )
        self.res_branch = ConvModule(out_channels,out_channels,1,1,norm_cfg=norm_cfg,act_cfg=None)
        self.channel_conv = ConvModule(out_channels*2,out_channels,1,1,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x):
        if self.down_sample!= None:
            x = self.down_sample(x)
        # 确定需要添加的填充量
        h, w = x.size(2), x.size(3)
        pad_h = (self.sp_num - h % self.sp_num) % self.sp_num  # 向上取整到最近的sp_num倍数
        pad_w = (self.sp_num - w % self.sp_num) % self.sp_num

        # 在高度和宽度上进行填充
        padding = (0, pad_w, 0, pad_h)  # (左, 右, 上, 下)
        x = F.pad(x, padding, mode=self.padding_mode, value=self.padding_value)

        x_1 = self.block_conv(x)

        # 在高度上分割
        height_chunks = torch.chunk(x, self.sp_num, dim=2)

        # 对每个高度子图再沿着宽度拆分
        processed_quarters = []
        for hc in height_chunks:
            width_chunks = torch.chunk(hc, self.sp_num, dim=3)
            processed_width_chunks = [self.block_conv(wc) for wc in width_chunks]
            # 沿着宽度维度拼接
            recombined_width_part = torch.cat(processed_width_chunks, dim=3)
            processed_quarters.append(recombined_width_part)

            # 沿着高度维度拼接
        recombined = torch.cat(processed_quarters, dim=2)
        out = torch.cat((x_1, recombined), 1)
        out = self.cat_conv(out)
        res_x = self.res_branch(x)
        res_x = torch.cat((out,res_x),1)

        return self.channel_conv(res_x)


'''------------------------------------------------------Net-------------------------------------------------------------'''

# @MODELS.register_module()
class SPNet(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 norm_eval: bool = False,  # 是否在评估模式下使用归一化，默认为false
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
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

        # out_channels = [32, 64, 128, 256, 512]
        out_channels = [64, 256, 512, 1024,2048]


        self.conv1 = stem_block(self.input_channel, out_channels[0],norm_cfg=norm_cfg,act_cfg=act_cfg)  #(bs,c,h,w)→(bs,32,h,w)

        self.stage1 = nn.Sequential(SPConv_Block(out_channels[0], out_channels[1],sp_num=4,down=True,norm_cfg=norm_cfg,act_cfg=act_cfg),
                                    SPConv_Block(out_channels[1], out_channels[1],sp_num=4,norm_cfg=norm_cfg,act_cfg=act_cfg))
        self.stage2 = nn.Sequential(SPConv_Block(out_channels[1], out_channels[2],sp_num=4,down=True,norm_cfg=norm_cfg,act_cfg=act_cfg),
                                    SPConv_Block(out_channels[2], out_channels[2],sp_num=4,norm_cfg=norm_cfg,act_cfg=act_cfg))
        self.stage3 = nn.Sequential(SPConv_Block(out_channels[2], out_channels[3],sp_num=2,down=True,norm_cfg=norm_cfg,act_cfg=act_cfg),
                                    SPConv_Block(out_channels[3], out_channels[3],sp_num=2,down=False,norm_cfg=norm_cfg,act_cfg=act_cfg),
                                    )
        self.stage4 = nn.Sequential(SPConv_Block(out_channels[3], out_channels[4],sp_num=2,down=True,norm_cfg=norm_cfg,act_cfg=act_cfg),
                                    SPConv_Block(out_channels[4], out_channels[4], sp_num=2, down=False, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                    )
        # self.aspp = ASPP(out_channels[4],[1, 6, 12, 18],out_channels[4],norm_cfg=norm_cfg, act_cfg=act_cfg)


    def forward(self, x):
        x0 = self.conv1(x)   # (bs,64,h,w)
        x1 = self.stage1(x0)  # (bs,256,h/2,w/2)
        x2 = self.stage2(x1)  # (bs,512,h/4,w/4)
        x3 = self.stage3(x2)  # (bs,1024,h/8,w/8)
        x4 = self.stage4(x3)  # (bs,2048,h/16,w/16)
        # x = self.aspp(x)    # (bs,2048,h/16,w/16)

        return x4



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
    x = torch.randn(2,3,256,256).cuda()
    moduel = SPNet(3,8).cuda()
    y = moduel(x)
    print(y.shape)

















