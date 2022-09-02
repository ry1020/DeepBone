import torch
from torch import nn

from models_2d.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def generate_model(opt):
    assert opt.model in [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2"
    ]

    if opt.model == 'resnet18':
        model = resnet18 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnet34':
        model = resnet34 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnet50':
        model = resnet50 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnet101':
        model = resnet101 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnet152':
        model = resnet152 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnext50_32x4d':
        model = resnext50_32x4d (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnext101_32x8d':
        model = resnext101_32x8d (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'resnext101_64x4d':
        model = resnext101_64x4d (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'wide_resnet50_2':
        model = wide_resnet50_2 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)
    elif opt.model == 'wide_resnet101_2':
        model = wide_resnet101_2 (zero_init_residual=opt.zero_init_residual,
                                      replace_stride_with_dilation=opt.replace_stride_with_dilation)

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
