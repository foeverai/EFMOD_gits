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
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

'''------------------------------------------------------------------------'''
# class Dif_att(BaseModule):
#
#     def __init__(self,
#                  in_chan: int,
#                  out_chan: int,
#                  norm_cfg= dict(type='BN'),
#                  act_cfg= dict(type='ReLU'),
#                  init_cfg=None,
#                  ):
#         self.in_chan = in_chan
#         self.out_chan = out_chan
#         self.






















