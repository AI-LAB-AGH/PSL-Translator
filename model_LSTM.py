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
        self.h_l = self.c_l = self.h_r = self.c_r = None

    def initialize_cell_and_hidden_state(self) -> None:
        """
        Initializes the cell state and hidden state for both hand modules with zeros.
        
        Should be called once before passing in each training sample, and once before
        starting real-time inference.
        
        """

        self.h_l = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial left hidden state
        self.c_l = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial left cell state
        self.h_r = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial right hidden state
        self.c_r = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial right cell state

    def split_input(self, input):
        body = input[0][:17, :]
        face = input[0][23:91, :]
        left = input[0][91:112, :]
        right = input[0][112:, :]

        source = input[0][0]
        body -= source
        face -= source
        left -= source
        right -= source

        body = body.view(17 * 2)
        face = face.view(68 * 2)
        left = left.view(21 * 2)
        right = right.view(21 * 2)

        return (left, right, face, body)

    def forward(self, left: torch.tensor, right: torch.tensor) -> torch.tensor:
        """
        Forwards a single frame of landmarks through the network. 
        
        Dimensions are annotated for a better understanding of how data is passed forward.
        Below is a dictionary for the symbols used:
            - L:   length of the sequence (always 1 in this version of the model)
            - N:   batch size
            - IN:  input size
            - H:   hidden size
            - LAY: number of layers in nn.LSTM
            - OUT: number of classes

        """
        
        # Input dims: N x L x IN
        if left.shape[2] != 0: # Left hand detected
            # After permute: L x N x IN
            left = torch.permute(left, (1, 0, 2))

            # Output dims: L x N x H
            out_l, (self.h_l, self.c_l) = self.lstm_l(left, (self.h_l, self.c_l))

            # Grab the output from the last timestep 
            out_l = out_l[-1, :, :]
        else:
            out_l = torch.zeros([left.shape[0], self.hidden_size])

        if right.shape[2] != 0: # Right hand detected
            right = torch.permute(right, (1, 0, 2))
            out_r, (self.h_r, self.c_r) = self.lstm_r(right, (self.h_r, self.c_r))
            out_r = out_r[-1, :, :]
        else:
            out_r = torch.zeros([right.shape[0], self.hidden_size])

        # Concatenate the left and right outputs
        out = torch.cat([out_l, out_r], dim=1)

        # Matmul dims: (N x 2H) * (2H x OUT) = N x OUT
        return self.fc(out)