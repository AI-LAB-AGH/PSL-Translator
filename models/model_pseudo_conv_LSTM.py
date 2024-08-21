from models.model_LSTM import LSTMModel, PseudoLSTMModel
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
        # TODO: Fix dimensionality (permute and flatten / unflatten batch and time series dimensions)
        # 1 x 480 x 640 x 2
        x = torch.permute(x, (0, 3, 1, 2))
        # B x C x H x W
        # 1 x 2 x 480 x 640
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
        x = x[None, :]
        x = torch.flatten(x, start_dim=2)
        # 1 x 1 x 2400
        return x


class ConvLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, device):
        super(ConvLSTM, self).__init__()
        self.conv = ConvModule().to(device)
        self.LSTM = PseudoLSTMModel(input_size=2400, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)
    
    def initialize_cell_and_hidden_state(self):
        self.LSTM.initialize_cell_and_hidden_state()

    def forward(self, x):
        x = self.conv(x)
        x = self.LSTM(x)
        return x
