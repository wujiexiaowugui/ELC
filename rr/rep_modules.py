from concurrent.futures import thread
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rr.rep_convert import fuse_conv_and_bn, fuse_conv_and_conv1x, fuse_conv1x_and_conv,fuse_shortcut, weight_autopad
from copy import deepcopy

from models.common import Conv, Conv_bn
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class RepReLU(nn.Module):
    def __init__(self, init: torch.Tensor):
        super(RepReLU, self).__init__()
        self.weight = init
        
    def forward(self, x):
        return F.prelu(x, 1.0 - self.weight)


class remrelu(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weight = init
        self.relu_alpha = nn.Parameter(torch.Tensor([1.0]))
        self.act = RepReLU(self.relu_alpha)
        self.rep_flag = [False]
        self.rep_keys = [self.relu_alpha]

    def forward(self, x):
        return self.act(x)

    def fuse(self, thershold=0.01, msg=False):
        
        if self.relu_alpha < thershold:  # switch to identity
            self.act = nn.Identity()
        else:
            self.act = nn.LeakyReLU(float(1 - self.relu_alpha.data))
        self.__delattr__('relu_alpha')
        self.rep_flag = [True]

        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.act(x)


class RepConv(nn.Module):
    # reparams convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # assert s == 1
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.conv1x = nn.Conv2d(c1, c2, 1, s, groups=g, bias=False)
        if g != 1:
            identity_mat = np.eye(max(c1, c2), dtype=np.float32) + 1e-6
            identity_mat = identity_mat[:c2, :1, None, None]
        else:
            identity_mat = np.eye(max(c1, c2), dtype=np.float32) + 1e-6
            identity_mat = identity_mat[:c2, :c1, None, None]
        self.conv1x.weight.data.copy_(torch.from_numpy(identity_mat))
        self.bn1x = nn.BatchNorm2d(c2)

        self.conv_alpha = nn.Parameter(torch.Tensor([1.0]))
        self.relu_alpha = nn.Parameter(torch.Tensor([1.0]))
        if act:
            self.act = RepReLU(self.relu_alpha)
        else:
            self.act = nn.Identity()

        self.kernel_size = k
        self.rep_keys = [[self.conv_alpha], [], [self.relu_alpha]]
        self.rep_flag = [True, True]  # conv-head and conv-end


    def forward(self, x):  # init: alpha=1
        x1 = self.bn(self.conv(x))
        x2 = self.bn1x(self.conv1x(x))
        # import pdb
        # pdb.set_trace()
        x = self.conv_alpha * x1 + (1 - self.conv_alpha) * x2
        x = self.act(x)
        return x

    def set_rep(self, head=None, tail=None):
        if head is not None:
            self.rep_flag[0] = head
        if tail is not None:
            self.rep_flag[-1] = tail

    def fuse(self, thershold=0.01, msg=False):
        self.conv = fuse_conv_and_bn(self.conv, self.bn)
        self.__delattr__('bn')
        self.conv1x = fuse_conv_and_bn(self.conv1x, self.bn1x)
        self.__delattr__('bn1x')

        if self.conv_alpha < thershold:  # switch to 1x1conv
            self.conv = self.conv1x
            self.kernel_size = 1

        else:  # merge two conv
            
            weight_fuse = self.conv_alpha * self.conv.weight + \
                          (1 - self.conv_alpha) * weight_autopad(self.conv1x.weight, self.conv.weight.shape)
            bias_fuse = self.conv_alpha * self.conv.bias + (1 - self.conv_alpha) * self.conv1x.bias
            self.conv.weight.data.copy_(weight_fuse)
            self.conv.bias.data.copy_(bias_fuse)
        self.__delattr__('conv1x')
        # print(self.relu_alpha)
        if self.relu_alpha < thershold:  # switch to identity
            self.act = nn.Identity()
        else:
            self.act = nn.LeakyReLU(float(1 - self.relu_alpha.data))
        # print(self.relu_alpha, self.act)
        self.set_rep(head=bool(self.kernel_size == 1))  # if conv-kernel-size == 1 ,could be attached at head
        self.set_rep(tail=bool(self.relu_alpha < thershold))  # if act is similar to y=x ,could be attached

        self.__delattr__('conv_alpha')
        self.__delattr__('relu_alpha')

        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class RepConv_noRelu(nn.Module):
    # reparams convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        # assert s == 1
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.conv1x = nn.Conv2d(c1, c2, 1, s, groups=g, bias=False)
        identity_mat = np.eye(max(c1, c2), dtype=np.float32) + 1e-6
        identity_mat = identity_mat[:c2, :c1, None, None]
        self.conv1x.weight.data.copy_(torch.from_numpy(identity_mat))
        self.bn1x = nn.BatchNorm2d(c2)

        self.conv_alpha = nn.Parameter(torch.Tensor([1.0]))
       

        self.kernel_size = k
        self.rep_keys = [[self.conv_alpha], [], [self.conv_alpha]]
        self.rep_flag = [True]  # conv-head and conv-end


    def forward(self, x):  # init: alpha=1
        x1 = self.bn(self.conv(x))
        x2 = self.bn1x(self.conv1x(x))
        x = self.conv_alpha * x1 + (1 - self.conv_alpha) * x2
        # x = self.act(x)
        return x

    def set_rep(self, head=None, tail=None):
        if head is not None:
            self.rep_flag[0] = head
        if tail is not None:
            self.rep_flag[-1] = tail

    def fuse(self, thershold=0.01, msg=False):
        self.conv = fuse_conv_and_bn(self.conv, self.bn)
        self.__delattr__('bn')
        self.conv1x = fuse_conv_and_bn(self.conv1x, self.bn1x)
        self.__delattr__('bn1x')

        if self.conv_alpha < thershold:  # switch to 1x1conv
            self.conv = self.conv1x
            self.kernel_size = 1

        else:  # merge two conv
            weight_fuse = self.conv_alpha * self.conv.weight + \
                          (1 - self.conv_alpha) * weight_autopad(self.conv1x.weight, self.conv.weight.shape)
            bias_fuse = self.conv_alpha * self.conv.bias + (1 - self.conv_alpha) * self.conv1x.bias
            self.conv.weight.data.copy_(weight_fuse)
            self.conv.bias.data.copy_(bias_fuse)
        self.__delattr__('conv1x')

        self.set_rep(head=bool(self.kernel_size == 1))  # if conv-kernel-size == 1 ,could be attached at head
        self.set_rep(tail=bool(self.kernel_size == 1))  # if act is similar to y=x ,could be attached

        self.__delattr__('conv_alpha')
        # self.__delattr__('relu_alpha')

        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.conv(x)




class RepResblock(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, stride=1, shortcut=True, g=1, e=1):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        self.cv1 = RepConv(c1, c_, 3, s=stride)
        self.cv2 = RepConv_noRelu(c_, c2, 3, 1, g=g)
       
        self.act = remrelu()
        self.downsample = nn.Identity()
        self.ds = False
        if stride != 1 or c1 != c2:
            self.ds = True
            self.downsample = Conv_bn(c1, c2, 1, s=stride)
        
        self.rep_keys = [self.cv1.conv_alpha, self.cv1.relu_alpha, self.cv2.conv_alpha, self.act.relu_alpha]
        self.rep_flag = [False, False]  
        self.fuse_shortcut_flag = 0

    def forward(self, x):
        skip = self.downsample(x)
        x = self.cv1(x)
        x = self.cv2(x)
        x = skip + x 
        out = self.act(x)
        return out

    def set_rep(self, head=None, tail=None):
        if head is not None:
            self.rep_flag[0] = head
        if tail is not None:
            self.rep_flag[-1] = tail

        # if self.add:
        if False in self.rep_flag:
            self.cv1.set_rep(head=False)
            self.act.rep_flag[-1]=False
       
    def fuse(self, thershold=0.01, msg=False):
        
        self.cv1.fuse(thershold)  # merge RepConv first
        self.cv2.fuse(thershold)
        self.act.fuse(thershold)
        if self.act.rep_flag[0] == False:
            self.rep_flag[-1] = False
        else:
            if self.cv1.rep_flag[-1] & self.cv2.rep_flag[0]: 
                self.cv1.conv = fuse_conv_and_conv1x(self.cv1.conv, self.cv2.conv, msg)
                self.cv2.conv = nn.Identity()
                self.cv1 = fuse_shortcut(self.cv1, self.downsample, ds=self.ds)
                self.fuse_shortcut_flag = 1
                self.forward = self.forword_fuse1
                self.rep_flag[-1] = True
                self.rep_flag[0] = self.cv1.rep_flag[0]
                
        rep_total_flag = False
        return rep_total_flag
    def forword_fuse1(self, x):
        return self.act(self.cv1(x))

