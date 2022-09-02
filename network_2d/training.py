import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))

    model.train()

    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()

    # end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        #data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = torch.sqrt(criterion(torch.squeeze(outputs), torch.squeeze(targets)))

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #batch_time.update(time.time() - end_time)
        #end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         #batch_time=batch_time,
                                                         #data_time=data_time,
                                                         loss=losses))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()


    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
