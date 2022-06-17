import argparse
from pathlib import Path


# Training settings
def parse_opts():
    parser = argparse.ArgumentParser(description='Bone Strength Project')
    parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_false', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

    args = parser.parse_args()

    return args
