from __future__ import print_function
import os
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

import inference
from training import train_epoch
from utils import Logger, worker_init_fn, get_lr
from validation import val_epoch
from model import generate_model, make_data_parallel
from data import get_training_data, get_inference_data, get_validation_data
from opts import parse_opts


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()
    
    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.n_input_channels = 1

    print(opt)
    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def resume_train_utils(resume_path, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler

def get_train_utils(opt, model_parameters):
    train_data = get_training_data(opt.data_path, opt.no_sim, opt.noise_scales, opt.resolution_scales, 
                                    opt.voxel_size_simulated)

    train_loader = DataLoader(dataset=train_data,
                              num_workers=opt.n_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    return train_loader, train_logger, train_batch_logger, optimizer, scheduler


def get_val_utils(opt):
    valid_data = get_validation_data(opt.data_path, opt.no_sim, opt.noise_scales, opt.resolution_scales, 
                                        opt.voxel_size_simulated)

    val_loader = DataLoader(valid_data,
                            batch_size=(opt.batch_size),
                            shuffle=False,
                            num_workers=opt.n_threads,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    inference_data = get_inference_data(opt.data_path, opt.no_sim, opt.noise_scales, opt.resolution_scales,
                                        opt.voxel_size_simulated, opt.manual_seed, opt.inference_subset)

    inference_loader = DataLoader(inference_data,
                                  batch_size=opt.inference_batch_size,
                                  shuffle=False,
                                  num_workers=opt.n_threads,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn)

    return inference_loader


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)

    model = make_data_parallel(model, opt.distributed, opt.device)
    parameters = model.parameters()

    if opt.is_master_node:
        print(model)

    criterion = MSELoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    if opt.inference:
        inference_loader = get_inference_utils(opt)

        inference_result_path = os.path.join(opt.result_path, '_'.join((opt.inference_subset, 'inference.csv')))

        inference.inference(inference_loader, model, inference_result_path)


if __name__ == '__main__':

    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True

    main_worker(opt)
