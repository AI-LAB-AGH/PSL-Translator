import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        # L     - seq_len
        # N     - batch_size
        # IN    - input_size
        # H     - hidden_size
        # LAY   - num_layers
        # OUT   - num_classes

        # Input dims: L x N x IN
        x = torch.permute(x, (1, 0, 2))

        # Cell and hidden state dims: LAY x N x H
        h_0 = torch.zeros([self.num_layers, x.shape[1], self.hidden_size]) # Initial hidden state
        c_0 = torch.zeros([self.num_layers, x.shape[1], self.hidden_size]) # Initial cell state

        # Output dims: L x N x H
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # Grab the output from the last timestep 
        out = out[-1, :, :]

        # Matmul dims: (N x H) * (H x OUT) = N x OUT
        return self.fc(out)