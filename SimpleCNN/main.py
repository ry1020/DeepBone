from __future__ import print_function

from math import log10

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_data, get_inference_data, get_validation_data
from opts import parse_opts


def train(epoch, training_data_loader, model, optimizer, criterion):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(opt.device), batch[1].to(opt.device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(torch.squeeze(output), torch.squeeze(target))
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def validate(val_data_loader, model, criterion):
    avg_mse = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input, target = batch[0].to(opt.device), batch[1].to(opt.device)
            prediction = model(input)
            mse = criterion(torch.squeeze(prediction), torch.squeeze(target))
            avg_mse += mse
    print("===> Avg. MSE: {:.4f}".format(avg_mse / len(val_data_loader)))


def checkpoint(epoch, model):
    model_out_path = "./output/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main_worker(opt):
    torch.manual_seed(opt.seed)

    opt.device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')
    train_set = get_training_data(opt.data_path)
    val_set = get_validation_data(opt.data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                 shuffle=False)

    print('===> Building model')
    model = Net().to(opt.device)
    criterion = nn.MSELoss().to(opt.device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch, training_data_loader, model, optimizer, criterion)
        validate(val_data_loader, model, criterion)
        checkpoint(epoch, model)


if __name__ == '__main__':

    opt = parse_opts()

    print(torch.cuda.is_available())

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    main_worker(opt)
