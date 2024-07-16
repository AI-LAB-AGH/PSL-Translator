import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm_l = nn.LSTM(input_size, hidden_size, num_layers)
        self.lstm_r = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, left: torch.tensor, right: torch.tensor) -> torch.tensor:
        # L     - seq_len
        # N     - batch_size
        # IN    - input_size
        # H     - hidden_size
        # LAY   - num_layers
        # OUT   - num_classes

        # Input dims: N x L x IN
        if left.dim() == 3:
            # After permute: L x N x IN
            left = torch.permute(left, (1, 0, 2))

            # Cell and hidden state dims: LAY x N x H
            h_0 = torch.zeros([self.num_layers, left.shape[1], self.hidden_size])  # Initial hidden state
            c_0 = torch.zeros([self.num_layers, left.shape[1], self.hidden_size])  # Initial cell state
            
            # Output dims: L x N x H
            out_l, _ = self.lstm_l(left, (h_0, c_0))

            # Grab the output from the last timestep 
            out_l = out_l[-1, :, :]
        else:
            out_l = torch.zeros([left.shape[0], self.hidden_size])

        if right.dim() == 3:
            right = torch.permute(right, (1, 0, 2))

            h_0 = torch.zeros([self.num_layers, right.shape[1], self.hidden_size])
            c_0 = torch.zeros([self.num_layers, right.shape[1], self.hidden_size])
            
            out_r, _ = self.lstm_r(right, (h_0, c_0))
            out_r = out_r[-1, :, :]
        else:
            out_r = torch.zeros([right.shape[0], self.hidden_size])

        # Concatenate the left and right outputs
        out = torch.cat([out_l, out_r], dim=1)

        # Matmul dims: (N x 2H) * (2H x OUT) = N x OUT
        return self.fc(out)