from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
# from .imagenet_ import get_imagenet_dataloaders, get_imagenet_dataloaders_sample
from .imagenet import Data

def get_dataset(cfg):
    sampler = None
    if cfg.dataset_type == "cifar100":
        
        train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.batch_size,
                val_batch_size=cfg.batch_size,
                num_workers=2,
            )
        num_classes = 100
    elif cfg.dataset_type == "cifar10":
        
        train_loader, val_loader, num_data = get_cifar10_dataloaders(
                batch_size=cfg.batch_size,
                val_batch_size=cfg.batch_size,
                num_workers=2,
            )
        num_classes = 10
    elif cfg.dataset_type == "imagenet":
        data_tmp = Data(cfg)
        train_loader = data_tmp.trainLoader
        val_loader = data_tmp.testLoader
        num_data = len(train_loader)
        # train_loader, val_loader, num_data, sampler = get_imagenet_dataloaders(
        #         batch_size=cfg.batch_size,
        #         val_batch_size=cfg.batch_size,
        #         num_workers=cfg.workers,
        #     )
        num_classes = 1000
    else:
        # print("****")
        raise NotImplementedError(cfg.dataset_type)

    return train_loader, val_loader, num_data, num_classes
