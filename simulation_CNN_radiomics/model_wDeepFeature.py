import torch
from torch import nn

from models_wDeepFeature import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet


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
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_input_channels=opt.n_input_channels,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      shortcut_type=opt.resnet_shortcut,
                                      widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth,
                                          n_input_channels=opt.n_input_channels,
                                          conv1_t_size=opt.conv1_t_size,
                                          conv1_t_stride=opt.conv1_t_stride,
                                          no_max_pool=opt.no_max_pool,
                                          shortcut_type=opt.resnet_shortcut,
                                          widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt.model_depth,
            n_input_channels=opt.n_input_channels,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
            shortcut_type=opt.resnet_shortcut,
            k=opt.wide_resnet_k)
    elif opt.model == 'resnext':
        model = resnext.generate_model(model_depth=opt.model_depth,
                                       n_input_channels=opt.n_input_channels,
                                       conv1_t_size=opt.conv1_t_size,
                                       conv1_t_stride=opt.conv1_t_stride,
                                       no_max_pool=opt.no_max_pool,
                                       shortcut_type=opt.resnet_shortcut,
                                       cardinality=opt.resnext_cardinality)
    elif opt.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.model_depth,
            n_input_channels=opt.n_input_channels,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
            shortcut_type=opt.resnet_shortcut)
    elif opt.model == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)

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
