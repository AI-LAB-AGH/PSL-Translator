from models.model_LSTM import PseudoLSTMModel
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
        x = torch.permute(x, (0, 3, 1, 2))
        # 2 x 480 x 640
        print(x.shape)
        print(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # 2 x 240 x 320
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # 2 x 120 x 160
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        # 2 x 60 x 80
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        # 2 x 30 x 40
        x = torch.flatten(x, 1)
        # 2400
        return x


class ConvLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()
        self.conv = ConvModule()
        self.LSTM = PseudoLSTMModel(input_size=1000, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    
    def initialize_cell_and_hidden_state(self):
        self.LSTM.initialize_cell_and_hidden_state()

    def forward(self, x):
        x = self.conv(x)
        x = self.LSTM(x)
        return x
