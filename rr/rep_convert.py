import torch
import torch.nn as nn
import torch.nn.functional as F
def fuse_shortcut(cv1, downsample, ds=False, msg=False):
    if ds:
        downsample.conv = fuse_conv_and_bn(downsample.conv, downsample.bn)
        weight_fuse_shortcut = cv1.conv.weight + weight_autopad(downsample.conv.weight, cv1.conv.weight.shape)
        bias_fuse_shortcut = cv1.conv.bias +  downsample.conv.bias
        cv1.conv.weight.data.copy_(weight_fuse_shortcut)
        cv1.conv.bias.data.copy_(bias_fuse_shortcut)                
        # downsample = nn.Identity()
                
    else:
        no, ni, _, _ = cv1.conv.weight.shape
        assert ni == no
        weight_shortcut = torch.eye(ni).view(ni, ni, 1, 1).to(cv1.conv.weight.device)
        weight_fuse = cv1.conv.weight + weight_autopad(weight_shortcut, cv1.conv.weight.shape)
        bias_fuse = cv1.conv.bias
        cv1.conv.weight.data.copy_(weight_fuse)
        cv1.conv.bias.data.copy_(bias_fuse)  # merge to cv1
                # self.add = False
    if msg:
        print('Removed one ShortcutLayer')
    return cv1

def weight_autopad(weight, target_weight_shape):
    diff = (target_weight_shape[-1] - weight.shape[-1])
    pad = (diff + 1) // 2
    assert pad >= 0
    return nn.functional.pad(weight, [diff - pad, pad, diff - pad, pad])
    # return nn.functional.pad(weight, [pad, diff - pad, pad, diff - pad])

def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).to(conv.weight.device)  # .requires_grad_(False)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuse_weight = torch.mm(w_bn, w_conv).view(fusedconv.weight.shape)
    fusedconv.weight.copy_(fuse_weight)

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuse_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    fusedconv.bias.copy_(fuse_bias)

    return fusedconv


def fuse_conv_and_conv1x(conv, conv1x, msg=False):
    # only available for  conv3*3->conv1*1 or conv1*1->conv1*1
    # conv1 : y1=w1*x+b1
    # conv2 : y2=w2*x+b2
    # fusedconv : y=w2*(w1*x+b1)+b2 = (w2*w1)*x+(w2*b1+b2)
    stride = tuple([conv.stride[i] * conv1x.stride[i] for i in range(len(conv.stride))])
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv1x.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).to(conv.weight.device).requires_grad_(False)
    w_conv3 = conv.weight
    w_conv1 = conv1x.weight
    fuse_weight = F.conv2d(w_conv3.permute(1, 0, 2, 3), w_conv1).permute(1, 0, 2, 3)
    fusedconv.weight.copy_(fuse_weight)

    b_conv3 = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_conv1 = torch.zeros(conv1x.weight.size(0), device=conv1x.weight.device) if conv1x.bias is None else conv1x.bias
    fuse_bias = F.conv2d(b_conv3[None, :, None, None], w_conv1).view(b_conv1.shape) + b_conv1

    fusedconv.bias.copy_(fuse_bias)

    if msg:
        print('Removed one ConvLayer')
    return fusedconv




def fuse_conv1x_and_conv(conv1x, conv, msg=False):
    # only available for  conv3*3->conv1*1 or conv1*1->conv1*1
    # conv1 : y1=w1*x+b1
    # conv2 : y2=w2*x+b2
    # fusedconv : y=w2*(w1*x+b1)+b2 = (w2*w1)*x+(w2*b1+b2)
    stride = tuple([conv1x.stride[i] * conv.stride[i] for i in range(len(conv1x.stride))])
    fusedconv = nn.Conv2d(conv1x.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).to(conv1x.weight.device).requires_grad_(False)
    w_conv1 = conv1x.weight
    w_conv3 = conv.weight
    repeat_size = w_conv3.shape[-1] + (w_conv3.shape[-1] - 1)
    fuse_weight = F.conv2d(w_conv1.permute(1, 0, 2, 3).repeat(1, 1, repeat_size, repeat_size),
                           w_conv3).permute(1, 0, 2, 3)
    fusedconv.weight.copy_(fuse_weight)

    b_conv1 = torch.zeros(conv1x.weight.size(0), device=conv1x.weight.device) if conv1x.bias is None else conv1x.bias
    b_conv3 = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    repeat_size = w_conv3.shape[-1]
    fuse_bias = F.conv2d(b_conv1[None, :, None, None].repeat(1, 1, repeat_size, repeat_size),
                         w_conv3).view(b_conv3.shape) + b_conv3

    fusedconv.bias.copy_(fuse_bias)

    if msg:
        print('Removed one ConvLayer')
    return fusedconv

