import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward

class DTCWT_downsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch,j=3):
        super(DTCWT_downsample_layer, self).__init__()
        self.j = j
        self.wt = DTCWTForward(J=j, biort='near_sym_a', qshift='qshift_a')
        self.conv_ = nn.Sequential(nn.Conv2d(in_ch, in_ch,kernel_size=3, stride=1, padding=1,groups=in_ch),
                                   nn.BatchNorm2d(in_ch),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv1_1 = nn.Conv2d(in_ch, in_ch, kernel_size=1,stride=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        """
        yl
        ([1,3,16,16])
        yh
        torch.Size([1, 3, 6, 32, 32, 2])
        torch.Size([1, 3, 6, 16, 16, 2])
        torch.Size([1, 3, 6, 8, 8, 2])
        """
        yL, yH = self.wt(x)
        print(yL.shape)
        # print(yL.shape)
        abs_ele = []
        num = len(yH)
        # 取模
        for i in range(num):
            real_part = yH[i][...,0]
            imag_part = yH[i][...,1]
            complex_y = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))
            abs_ele.append(torch.abs(complex_y))
        y_15 = abs_ele[0][:,:,0,::]
        y_45 = abs_ele[0][:,:,1,::]
        y_75 = abs_ele[0][:,:,2,::]
        y_105 = abs_ele[0][:,:,3,::]
        y_135 = abs_ele[0][:,:,4,::]
        y_165 = abs_ele[0][:,:,5,::]
        temp_out = self.conv1_1(y_15+y_45+y_75+y_105+y_135+y_165)
        # print(out.shape)
        out = self.upsample(self.conv_bn_relu(temp_out+self.conv1_1(yL)))
        return out