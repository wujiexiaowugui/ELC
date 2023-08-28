# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import imp
import json
from operator import mod
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import time
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# import logger
from utils.general import  check_requirements, set_logging,  print_args
from utils.torch_utils import  intersect_dicts
from dataset import get_dataset
from dataset.imagenet import Data
import thop
# from train import validate
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    flops, params = thop.profile(distiller, inputs=(image[0:1,:,:,:],), verbose=False)
    print('**flops: ', flops)
    print('**params: ', params)
    return top1.avg, top5.avg, losses.avg


@torch.no_grad()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='cifar10', help='dataset.yaml path')
    parser.add_argument('--profile', type=bool, default=True, help='display flops')
    parser.add_argument('--dataloader', type=str, default='cifar10', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='models/convnext.yaml', help='model.yaml path')
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolo5s-base/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--weights', type=str,
                        default='./runs/train/convnext/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    # parser.add_argument('--name', default='yolo5-base', help='save to project/name')
    parser.add_argument('--name', default='convnext', help='save to project/name')
    parser.add_argument('--exist-ok', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    # opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    # opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt

def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
def main(opt):
    set_logging()
    check_requirements(exclude=('tensorboard', 'thop'))
    if opt.dataset_type == 'imagenet':
        data_tmp = Data(opt)        
        val_dataloader = data_tmp.testLoader
        # val_dataloader = get_imagenet_val_loader(val_batch_size=1)
    elif opt.dataset_type == 'cifar100':
            train_loader, val_dataloader, num_data, num_classes = get_dataset(opt)
    elif opt.dataset_type == 'cifar10':
            train_loader, val_dataloader, num_data, num_classes = get_dataset(opt)
   
    ckpt = torch.load(opt.weights, map_location='cpu')
    model = ckpt['model'].cuda()
   
    csd = ckpt['model'].float().cuda().state_dict() 
   
    csd = intersect_dicts(csd, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=False)  # load
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items')  # report
    
   
    try:
        best_acc = torch.load(opt.weights, map_location='cuda')['best_fitness']
    except:
        best_acc = 0.

    # Configure
    model.eval()
    # print(model.rep_keys)
    top1, top5, loss = validate(val_loader=val_dataloader,distiller=model)
    
    
    print('top1: %s, top5: %s, best_acc: %s'%(top1.cpu().numpy(), top5.cpu().numpy(), best_acc.cpu().numpy()))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
