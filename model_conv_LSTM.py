from model_LSTM import LSTMModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # 640 x 400 x 2
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # 320 x 200 x 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # 160 x 100 x 2
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        # 80 x 50 x 2
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        # 40 x 25 x 2
        x = torch.flatten(x)
        # 2000
        return x


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv = ConvModule()
        self.LSTM = LSTMModel(input_size=1000)
    
    def forward(self, x):
        x = ConvModule(x)
        x = LSTMModel(x)
        return x
