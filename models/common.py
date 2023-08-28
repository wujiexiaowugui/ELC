# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import logging
from unittest import skip
import warnings
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from utils.general import colorstr, increment_path, save_one_box, xyxy2xywh
from utils.plots import Annotator, colors
from rr.rep_convert import fuse_conv_and_bn
LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
      
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() #if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Conv_bn(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse(self):
        self.conv = fuse_conv_and_bn(self.conv, self.bn)
        self.forward = self.forword_fuse

    def forword_fuse(self, x):
        return self.conv(x)


class Resblock(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2,  stride=1, e=1 ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2*e)
        self.c1 = c1
        self.c2 = c2
        self.downsample = nn.Identity()
        if stride != 1 or c1 != c2:
            self.downsample = Conv_bn(c1, c2, 1, s=stride)
            
        self.cv1 = Conv(c1, c_, 3, s=stride)
        self.cv2 = Conv_bn(c_, c2, 3, 1)
        self.act = nn.ReLU() 
    def forward(self, x):
        skip = self.downsample(x)
        # if self.c1 == self.c2 :
        return self.act(skip + self.cv2(self.cv1(x)))
        # else:
        #     return self.act( self.cv2(self.cv1(x)))

class InvertedResidualBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, e=6, shortcut=True, g=1):
        super(InvertedResidualBlock, self).__init__()
        c_ = c1 * e
        print("c1: %d, c_: %d, c2: %d, e: %d\n"%(c1, c_, c2, e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, s=stride, g=c_)
        self.cv3 = Conv_bn(c_, c2, 1, s=1, g=g)
        
        self.is_residual = True if stride == 1 else False
        self.is_conv_res = False if c1 == c2 else True
        if stride == 1 and self.is_conv_res:
            self.conv_res = Conv_bn(c1, c2 , 1, s=stride, g=g)
        # self.act = nn.ReLU() 
    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        out = self.cv3(self.cv2(self.cv1(x)))
        if self.is_residual:
            if self.is_conv_res:
                out = out + self.conv_res(x)
        return out#self.act(x) 

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

