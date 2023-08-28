# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import imp
from selectors import EpollSelector
# from stat import IO_REPARSE_TAG_MOUNT_POINT
import sys
from copy import deepcopy
from pathlib import Path
import math

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from models.common import *
from rr.rep_modules import *
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)




class RepModel(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        self.ch = ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
      
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        set_rep_flag(self.model)
        self.rep_keys = auto_select_keys(self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.yaml['channels'], self.yaml['ch'])

        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        for m in self.model:
               
            x = m(x)
                
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
       
        return x


    
    def check_fuse(self, module):
        try:
            in_channel = module.conv.in_channels
        except:
            in_channel = module.cv1.conv.in_channels

        input = torch.ones((1, in_channel, 320, 320)).cuda()
        with torch.no_grad():
            m = deepcopy(module)
            out = m(input)
            m.fuse()
            out_ = m(input)
            print(abs(out - out_).max())

    def is_empty(self, module):
        if sum(x.numel() for x in module.parameters()):
            return False
        else:
            return True

    def auto_remove(self, msg=False):
        modules = list(self.model.children())
        # special_layers = (SPP, SPPF, Concat, Detect, nn.Upsample)
        for i, module in enumerate(modules):

            if msg:
                print(module)
        self.model = nn.Sequential(*modules)

    def prune_info(self, model):
        count, count_preserved = 0, 0
        import pdb
        # pdb.set_trace()
        modules = list(self.model.children())
        for m in modules:
            # print(m)
            import pdb
            # pdb.set_trace()
            # if isinstance(m, RepConv):
            if isinstance(m, (Conv, RepConv)):
                count += 1
                if not self.is_empty(m):
                    count_preserved += 1
            if isinstance(m, (Resblock, RepResblock)):
                count += 2
                if not self.is_empty(m.cv1):
                    count_preserved += 1
                if not self.is_empty(m.cv2):
                    count_preserved += 1

        prune_rate = (count - count_preserved) * 1.0 / count
        print('Total RepConvs: ', count, ' Pruned: ', count - count_preserved,
              ' , Pruning rate : ', int(prune_rate * 100.0), ' %')
        return prune_rate

    def fuse(self, thershold=0.01, msg=False):  # fuse model Conv2d() + BatchNorm2d() layers
        if msg:
            LOGGER.info('Fusing layers... ')
        modules = list(self.model.children())
        special_layers = (nn.Upsample,  nn.MaxPool2d)
        for module in modules:  # fuse each module into one
            print(module.rep_keys)
            if msg:
                print(module.i)
            if not isinstance(module, special_layers):
               
                module.fuse(thershold, msg)
        for module in modules:
            if module.i == 0:
                pre = module
                continue
            cur = module
            
            if not isinstance(pre, special_layers) and not isinstance(cur, special_layers):
                    if pre.rep_flag[-1] & cur.rep_flag[0]:
                        try:
                            pre_conv = deepcopy(pre.conv)  # choose first conv and clear it
                            pre.conv = nn.Identity()
                        except:
                            pass
 
                        try:
                            cur.conv = fuse_conv_and_conv1x(pre_conv, cur.conv, msg)  # choose second conv
                        except:
                            # pass
                            if isinstance(pre, RepResblock) and isinstance(cur, RepResblock):
                                    if isinstance(pre.cv2.conv, nn.Identity):
                                        pre_conv = deepcopy(pre.cv1.conv)
                                        pre.cv1.conv = nn.Identity()
                                        print(pre_conv)
                                        print(cur.cv1.conv)
                                        cur.cv1.conv = fuse_conv_and_conv1x(pre_conv, cur.cv1.conv, msg)
                                        print('fuse')
                            elif isinstance(pre, RepConv) and isinstance(cur, RepResblock):
                                if isinstance(pre.conv, nn.Identity):
                                        pre_conv = deepcopy(pre.cv1.conv)
                                        pre.cv1.conv = nn.Identity()
                                        print(pre_conv)
                                        print(cur.cv1.conv)
                                        cur.cv1.conv = fuse_conv_and_conv1x(pre_conv, cur.cv1.conv, msg)
                                        print('fuse')
                            
            pre = cur
        if msg:
            self.prune_info()
            self.auto_remove(msg)
        self.info()
        return self
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)



def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    idx = 0
    nc, gd, gw =  d['nc'], d['depth_multiple'], d['width_multiple']
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = nc#na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        if m in ('Conv','Resblock'):
            m = 'Rep' + m
        m = eval(m) if isinstance(m, str) else m  # eval strings

        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
       
        if n > 1:
            for tmp in range(n):
                
                m_ = m(*args)  # module
                t = str(m)[8:-2].replace('__main__.', '')  # module type
                np = sum([x.numel() for x in m_.parameters()])  # number params
                m_.i, m_.f, m_.type, m_.np = idx, f, t, np  # attach index, 'from' index, type, number params
                LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (idx, f, n_, np, t, args))  # print
                save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
                layers.append(m_)
                idx = idx + 1
        else:
            
            m_ = m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = idx, f, t, np  # attach index, 'from' index, type, number params
            LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (idx, f, n_, np, t, args))  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            idx += 1
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def set_rep_flag(model):
    modules = list(model.children())
    special_layers = (Concat, nn.Upsample)
    for module in modules:  # set rep_flag
        if module.i == 0:  # skip for the first conv
            module.set_rep(head=False)
            continue
        if isinstance(module, special_layers):  # no rep neighbouring special layer
            for i in module.f if isinstance(module.f, list) else [module.f]:  # for multi-input
                idx = module.i - 1 if i == -1 else i  # no rep before special
                if not isinstance(modules[idx], special_layers):
                    modules[idx].set_rep(tail=False)
            if module.i + 1 < len(modules) and not isinstance(modules[module.i + 1], special_layers):
                modules[module.i + 1].set_rep(head=False)  # no rep after special



def MaxPool2d():
    return nn.MaxPool2d()
def auto_select_keys(model):  # search for each Conv
    from rr.rep_modules import rep_pairs_append
    modules = list(model.children())
    special_layers = (Concat,  nn.Upsample,  nn.MaxPool2d)
    print("len modules:",len(modules))
    rep_keys = list()
    for module in modules:
        print(module)
        import pdb
        
        if module.i == 0:
            pre = module
            rep_keys = []
            continue
        cur = module
        
        # if not isinstance(pre, special_layers) and not isinstance(cur, special_layers):
        if not isinstance(cur, special_layers):
            if isinstance(cur, RepResblock):
                    rep_keys.append([cur.rep_keys[0], cur.rep_keys[1]])
                    rep_keys.append([cur.rep_keys[2], cur.rep_keys[3]])
            else:
                    exit(0)
        pre = cur
   
    print('Total rep-keys:', len(rep_keys))
    return rep_keys






















