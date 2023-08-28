import torch

def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx, epoch, idx2mask=None, opt=None):
        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                # bn_module = module_list[idx][1]
                bn_module = module_list[idx][1] if type(
                    module_list[idx][1]).__name__ == 'BatchNorm2d' else module_list[idx][0]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
            if idx2mask:
                for idx in idx2mask:
                    # bn_module = module_list[idx][1]
                    bn_module = module_list[idx][1] if type(
                        module_list[idx][1]).__name__ == 'BatchNorm2d' else module_list[idx][0]
                    #bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.data.sub_(0.99 * s * torch.sign(bn_module.weight.data) * idx2mask[idx].cuda())

    @staticmethod
    def updateBN_scaler(sr_flag, module_list, s, prune_idx, epoch,scaler, idx2mask=None, opt=None):
        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                # bn_module = module_list[idx][1]
                bn_module = module_list[idx][1] if type(
                    module_list[idx][1]).__name__ == 'BatchNorm2d' else module_list[idx][0]
                bn_module.weight.grad.data.add_(scaler.scale(s * torch.sign(bn_module.weight.data)))  # L1
            if idx2mask:
                for idx in idx2mask:
                    # bn_module = module_list[idx][1]
                    bn_module = module_list[idx][1] if type(
                        module_list[idx][1]).__name__ == 'BatchNorm2d' else module_list[idx][0]
                    # bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.data.sub_(scaler.scale(0.99 * s * torch.sign(bn_module.weight.data) * idx2mask[idx].cuda()))

def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                #spp前一个CBL不剪 区分tiny
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'route' and 'groups' in module_defs[i+1]:
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'convolutional_nobias':
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'maxpool':
                # sppf前一个CBL不剪
                ignore_idx.add(i)
        elif module_def['type'] == 'convolutional_noconv':
            CBL_idx.append(i)
            ignore_idx.add(i)
        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

        elif module_def['type'] == 'upsample':
            #上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)


    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def parse_module_defs2(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                # spp前一个CBL不剪 区分spp和tiny
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'route' and 'groups' in module_defs[i + 1]:
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'maxpool':
                # sppf前一个CBL不剪
                ignore_idx.add(i)

        elif module_def['type'] == 'convolutional_noconv':
            CBL_idx.append(i)

        elif module_def['type'] == 'upsample':
            # 上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

        elif module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':

                # ignore_idx.add(identity_idx)
                shortcut_idx[i - 1] = identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':

                # ignore_idx.add(identity_idx - 1)
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all


def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] if type(module_list[idx][1]).__name__ == 'BatchNorm2d' else module_list[idx][0].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone() if type(module_list[idx][1]).__name__ == 'BatchNorm2d' else module_list[idx][0].weight.data.abs().clone()
        index += size

    return bn_weights

def check_pruned_rate(model):
    channels_total=0
    channels_pruned8e=0
    channels_pruned_e1=0
    channels_pruned_e2=0
    channels_pruned_e3=0
    for child_module in model.modules():
        if hasattr(child_module, 'conv_idx'):
            compactor_mat_ = child_module.pwc.weight.detach().clone()
            compactor_mat_ = (compactor_mat_ ** 2).sum(dim=(1, 2, 3))**0.5
            channels_total += len(compactor_mat_)
            channels_pruned8e += len(compactor_mat_[compactor_mat_ < 8e-1])
            channels_pruned_e1 += len(compactor_mat_[compactor_mat_ < 1e-1])
            channels_pruned_e2 += len(compactor_mat_[compactor_mat_ < 1e-2])
            channels_pruned_e3 += len(compactor_mat_[compactor_mat_ < 1e-3])
    print('Truth: have channels:',channels_total,', be pruned through 8e-1:',channels_pruned8e,'now could be pruned through 1e-1:',channels_pruned_e1,', be pruned through 1e-2:',channels_pruned_e2,', be pruned through 1e-3:',channels_pruned_e3)

def check_layer_pruned_rate(model):
    channels_total=0
    channels_pruned8e=0
    channels_pruned_e1=0
    channels_pruned_e2=0
    channels_pruned_e3=0
    for child_module in model.modules():
        if hasattr(child_module, 'mask'):
            weight = child_module.weight.detach().clone()
            channels_total += len(weight)
            channels_pruned8e += 1 if weight < 8e-1 else 0
            channels_pruned_e1 += 1 if weight < 1e-1 else 0
            channels_pruned_e2 += 1 if weight < 1e-2 else 0
            channels_pruned_e3 += 1 if weight < 1e-3 else 0
    print('Truth: have channels:',channels_total,', be pruned through 8e-1:',channels_pruned8e,'now could be pruned through 1e-1:',channels_pruned_e1,', be pruned through 1e-2:',channels_pruned_e2,', be pruned through 1e-3:',channels_pruned_e3)
