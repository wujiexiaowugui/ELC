# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""
import argparse
import imp
import random
import logging
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from dataset import get_dataset
import val  # for end-of-epoch mAP
from rr.rep_yolo import RepModel
from rr.rep_modules import RepConv
from models.common import Conv
from utils.general import  increment_path,  get_latest_run,  check_git_status, check_img_size, check_requirements, \
     check_yaml, check_suffix, print_args, print_mutation, set_logging,  colorstr, methods
from utils.downloads import attempt_download
from utils.torch_utils import  ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first, intersect_dicts1
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks
import torch.optim as optim

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
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
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
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
    return top1.avg, top5.avg, losses.avg

def adjust_learning_rate(epoch, optimizer, LR, LR_DECAY_STAGES, LR_DECAY_RATE, opt):
    if opt.dataset_type == 'imagenet':
        lr_policy = 'cos'
    else:
        lr_policy = 'step'

    if lr_policy == 'step':
        steps = np.sum(epoch > np.asarray(LR_DECAY_STAGES))
        if steps > 0:
            new_lr = LR * (LR_DECAY_RATE**steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            return new_lr
    elif lr_policy == 'cos':  # cos with warm-up
        new_lr = 0.5 * LR * (1 + math.cos(math.pi * (epoch - 5) / (opt.epochs - 5)))
        for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
        return new_lr
    return LR


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    cuda = device.type != 'cpu'
   
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    official=False
    if pretrained:
        print("*** loading best ckpt", weights)
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        import pdb
        # pdb.set_trace()
        
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = RepModel(cfg or ckpt['model'].yaml, ch=3, nc=None, anchors=hyp.get('anchors')).to(device)  # create
        exclude = None  # exclude keys
        try:
            csd1 = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd1, model.state_dict(), exclude=exclude)  # intersect
        except:
            csd = ckpt#torch.load_state_dict(weights)
            csd = intersect_dicts1(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} of {len(csd1)} items from {weights}')  # report
    else:
        model = RepModel(cfg, ch=3, nc=None).to(device)  # create

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    nbs = opt.batch_size  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2, g_all = [], [], [], []  # optimizer parameter groups
    for key, value in model.named_parameters():
        g_all.append(value)
        if not value.requires_grad:
            continue
        if "bias" in key:
            g2.append(value)
        elif 'conv' in key and 'weight' in key:
            g1.append(value)
        else:
            g0.append(value)

   
    optimizer = optim.SGD(
                g_all,
                lr=opt.lr,
                momentum=hyp['momentum'],
                weight_decay=hyp['weight_decay'],
            )
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    
    # if pretrained:
    #     print('*** load optimizer and epoch')
    #     # Optimizer
    #     if ckpt['optimizer'] is not None:
    #         optimizer.load_state_dict(ckpt['optimizer'])
    #         best_fitness = ckpt['best_fitness']

    #     # EMA
    #     if ema and ckpt.get('ema'):
    #         ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
    #         ema.updates = ckpt['updates']

    #     # Epochs
    #     start_epoch = ckpt['epoch'] + 1
    #     if resume:
    #         assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
    #     if epochs < start_epoch:
    #         LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
    #         epochs += ckpt['epoch']  # finetune additional epochs

    #     del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, val_loader, num_data, num_classes = get_dataset(opt)
    nc = num_classes

    nb = len(train_loader)  # number of batches
    

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

   

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # last_opt_step = -1
    # maps = np.zeros(nc)  # mAP per class
    results = [0, 0] # (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # scheduler.last_epoch = start_epoch - 1  # do not move
    # stopper = EarlyStopping(patience=opt.patience)
    # compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    def cmp_sum(elem):
        return elem[0] + elem[1]
    test_acc, test_acc_top5, test_loss = validate(val_loader, model)
    prune_ratio = 0
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        
        lr = adjust_learning_rate(epoch, optimizer, opt.lr, opt.lr_decay_stages, opt.lr_decay_rate, opt=opt)

        model.train()
        # mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'cls'))
        # LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        import pdb
        # pdb.set_trace()
        
        for i, data in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            try:
                imgs, target, index = data
            except:
                imgs, target = data
            # imgs, target, index = data
            imgs = imgs.float()
            imgs = imgs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # index = index.cuda(non_blocking=True)
            prune_step = 512#nb*30
            if ni % prune_step == 0:  # prune for every 256 steps,total 55 Convs, 49 rep-keys
                # pdb.set_trace()
                layers_to_prune = ni // prune_step
                # import pdb
                # pdb
                layers_to_prune = min(layers_to_prune, int(len(model.module.rep_keys) * opt.prune_ratio))
                # layers_to_prune = min(layers_to_prune, int(57 * opt.prune_ratio))  # max prune 100% layers
                model.module.rep_keys.sort(key=cmp_sum)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
               

            pred = model(imgs)  # forward
            loss = F.cross_entropy(pred, target)#, loss_items = compute_loss(pred, target.to(device))  # loss scaled by batch_size
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.

            # Backward
            loss.backward()
            # random
            # if epoch % 5:  # free one epoch for every 5 epochs
            import pdb
            # pdb.set_trace()
            penalty = opt.prune_factor
            for idx in range(len(model.module.rep_keys)):
                    # if idx > 0:
                        if idx < layers_to_prune:  # with only pruning
                            for ele_num in range(len(model.module.rep_keys[idx])):
                                model.module.rep_keys[idx][ele_num].grad.data.copy_(penalty * torch.sign(model.module.rep_keys[idx][ele_num]))
                                
                        else:
                            for ele_num in range(len(model.module.rep_keys[idx])):
                                model.module.rep_keys[idx][ele_num].grad.data.add(penalty * torch.sign(model.module.rep_keys[idx][ele_num]))
                               


            # Optimize
            # if ni - epochs >= accumulate:
            optimizer.step()  # optimizer.step
            optimizer.zero_grad()
            if ema:
                ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                # mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 1) % (f'{epoch}/{epochs - 1}', mem, loss))
                pbar.update()
                # callbacks.run('on_train_batch_end', ni, model, imgs, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------
        count = []  # count pruned
        keys = np.array([])
        for key in model.module.rep_keys:
            keys = np.append(keys, float(key[0]+key[1]))
        count.append((keys < 0.001).sum())
        count.append((keys < 0.01).sum())
        count.append((keys < 0.1).sum())
        count.append((keys < 0.5).sum())
        count.append((keys < 0.8).sum())
        
        # model_ = deepcopy(model.module).eval()
        # with torch.no_grad():
        #     model_.fuse()
        #     prune_ratio = model_.prune_info(model_)
        # lr = [x['lr'] for x in optimizer.param_groups]  # for loggers

        print('RepKeys total:', len(keys),
              ',to prune:', layers_to_prune,
              ',prune factor:', opt.prune_factor,
              ',ratio:', int(prune_ratio * 100),
              '% ,pruned through 0.5:', count[3],
              ', 0.1:', count[2],
              ', 0.01:', count[1],
              ', 0.001:', count[0],
              'lr:', lr)
        
        rep_values = list()
        for rep_id in range(len(model.module.rep_keys)):
            rep_values_cur_layer = list()
            for ele_num in range(len(model.module.rep_keys[rep_id])):
                rep_values_cur_layer.append(model.module.rep_keys[rep_id][ele_num].data.cpu().numpy()[0])
                    # model.module.rep_keys[idx][ele_num].grad.data.copy_(penalty * torch.sign(model.module.rep_keys[idx][ele_num]))
            rep_values.append(rep_values_cur_layer)   
        print(rep_values)
        # Scheduler
        # scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) #or stopper.possible_stop
            test_acc, test_acc_top5, test_loss = validate(val_loader, model)
            import pdb 
            # pdb.set_trace()
            if test_acc > best_fitness:
                # print(test_acc)
                best_fitness = test_acc
                results[0] = test_acc
                results[1] = test_acc_top5
            log_vals = [test_acc.cpu().numpy()] + [test_acc_top5.cpu().numpy()] + [lr] + [int(prune_ratio*100)]
            callbacks.run('on_fit_epoch_end_prune', log_vals, epoch, best_fitness.cpu().numpy())

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'cur_acc': test_acc,
                        "prune_ratio": int(prune_ratio*100),
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'rep_keys': model.module.rep_keys,
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == test_acc:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % 50 == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, test_acc)

    torch.cuda.empty_cache()
    return results

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='')
    # /home/hnu/projects/pruning/yolo-pruning/cls/runs/train/vgg19/weights
    parser.add_argument('--weights', type=str, default='/home/hnu/projects/pruning/ELC/runs/train/convnext/weights/best.pt')
    parser.add_argument('--lr', type=float, default='0.01')
    # parser.add_argument('--lr_decay_stages', type=list, default=[120,160,220])
    parser.add_argument('--lr_decay_stages', type=list, default=[140,240])
    parser.add_argument('--name', default='resnet20_e300', help='save to project/name')
    parser.add_argument('--lr_decay_rate', type=list, default=0.1)
    parser.add_argument('--cfg', type=str, default='models/convnext.yaml', help='model.yaml path')
    parser.add_argument('--dataset_type', type=str, default='cifar10', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch_prune.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--prune_ratio', type=float, default=0.60)
    parser.add_argument('--prune_factor', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')    
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--exist-ok', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=600, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
    parser.add_argument('--seed', type=bool, default=False)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    set_logging(RANK)
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])
    if opt.seed:
        setup_seed(0)
    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        results = train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            print_mutation(tuple(results), hyp.copy(), save_dir, opt.bucket)

        # Plot results
        # plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    import shutil
    shutil.copyfile()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
