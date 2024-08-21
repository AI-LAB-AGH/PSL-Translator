import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.h = self.c = None


    def initialize_cell_and_hidden_state(self) -> None:
        """
        Initializes the cell state and hidden state for both hand modules with zeros.
        
        Should be called once before passing in each training sample, and once before
        starting real-time inference.
        
        """

        self.h = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial hidden state
        self.c = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial cell state


    def prepare_input(self, x: torch.tensor):
        x = x.float()
        x = x.unsqueeze(1)

        # Input dims: N x L x IN
        source_body = x[0][0][0].clone()
        source_face = x[0][0][53].clone()
        source_left = x[0][0][100].clone()
        source_right = x[0][0][121].clone()

        body = x[0][0][:17]
        feet = x[0][0][17:23]
        face = x[0][0][23:91]
        left = x[0][0][91:112]
        right = x[0][0][112:]
        w_left = torch.max(left[:, 0]) - torch.min(left[:, 0])
        h_left = torch.max(left[:, 1]) - torch.min(left[:, 1])
        w_right = torch.max(right[:, 0]) - torch.min(right[:, 0])
        h_right = torch.max(right[:, 1]) - torch.min(right[:, 1])

        chin2left = left - source_body
        chin2right = right - source_body
        body -= source_body
        face -= source_face
        left -= source_left
        right -= source_right

        if w_left != 0. and h_left != 0:
            left[:, 0] /= w_left
            left[:, 1] /= h_left

        if w_right != 0. and h_right != 0:
            right[:, 0] /= w_right
            right[:, 1] /= h_right

        x = torch.concat([body, feet, face, left, right, chin2left, chin2right])
        x = x.view(1, 1, self.input_size)

        return x
    

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.prepare_input(x)
        x = torch.permute(x, (1, 0, 2))

        out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        out = out[-1, :, :]
        return self.fc(out)


class PseudoLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PseudoLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h = None
        self.c = None

    def initialize_cell_and_hidden_state(self) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.h = torch.zeros([self.num_layers, 1, self.hidden_size]).to(device)  # Initial hidden state
        self.c = torch.zeros([self.num_layers, 1, self.hidden_size]).to(device)  # Initial cell state

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forwards a single frame of landmarks through the network. 
        
        Dimensions are annotated for a better understanding of how data is passed forward.
        Below is a dictionary for the symbols used:
            - L:   length of the sequence (always 1 in this version of the model)
            - N:   batch size
            - IN:  input size
            - H:   hidden size
            - OUT: number of classes

        """
        # Input dims: N x L x IN
        #x = torch.permute(x, (1, 0, 2))

        # Output dims: L x N x H
        out, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        # Grab the output from the last timestep 
        out = out[-1, :, :]

        # Matmul dims: (N x H) * (H x OUT) = N x OUT
        return self.fc(out)