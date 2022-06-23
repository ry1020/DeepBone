import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm3d(256)

        self.max_pool = nn.MaxPool3d(kernel_size=3, padding=1, stride=2)
        self.act = nn.ReLU()

        self.avg_pool = nn.AvgPool3d(kernel_size=3, padding=1, stride=2)
        self.flat = nn.Flatten()

        self.linear = nn.Linear(256 * 8 * 8 * 8, 256)
        self.norm5 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(256, 1)

        self._initialize_weights()

    def forward(self, x):
        # print("got input size {}".format(x.size()))
        x = self.act(self.conv1(x))  # [batch_size, 32, 256, 256, 256]
        x = self.norm1(self.max_pool(x))  # [batch_size, 32, 128, 128, 128]
        x = self.act(self.conv2(x))  # [batch_size, 64, 128, 128, 128]
        x = self.norm2(self.max_pool(x))  # [batch_size, 64, 64, 64, 64]
        x = self.act(self.conv3(x))  # [batch_size, 128, 64, 64, 64]
        x = self.norm3(self.max_pool(x))  # [batch_size, 128, 32, 32, 32]
        x = self.act(self.conv4(x))  # [batch_size, 256, 32, 32, 32]
        x = self.norm4(self.max_pool(x))  # [batch_size, 256, 16, 16, 16]
        x = self.avg_pool(x)  # [batch_size, 256, 8, 8, 8]
        x = self.flat(x) # [batch_size, 256*8*8*8=131072]
        x = self.act(self.linear(x))  # [batch_size, 256]
        # print("got input size {}".format(x.size()))
        x = self.norm5(x)
        x = self.drop(x)
        x = self.out(x)  # [batch_size, 1]
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
