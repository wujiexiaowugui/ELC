import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
from dataset.imagenet_dateset import ColorAugmentation
data_path = '/home/hnu/projects/datasets/imageNet'
class Data:
    def __init__(self, args):
        pin_memory = True
        # if args.device is not None:
        #     pin_memory = True
        scale_size = 224

        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                ColorAugmentation(),
                normalize,
            ]))
        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=pin_memory,
        )
        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))
        self.testLoader = DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
