import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.l = nn.LSTM(21 * 2, hidden_size, num_layers)
        self.r = nn.LSTM(21 * 2, hidden_size, num_layers)
        self.f2l = nn.LSTM(21 * 2, hidden_size, num_layers)
        self.f2r = nn.LSTM(21 * 2, hidden_size, num_layers)
        self.h2h = nn.LSTM(21 * 2, hidden_size, num_layers)
        
        self.fuse = nn.Linear(5*hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.h_l = self.c_l = self.h_r = self.c_r = self.h_f2r = self.c_f2l = self.h_h2h = self.c_h2h = None


    def initialize_cell_and_hidden_state(self) -> None:
        """
        Initializes the cell state and hidden state for both hand modules with zeros.
        
        Should be called once before passing in each training sample, and once before
        starting real-time inference.
        
        """

        self.h_l = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial hidden state
        self.c_l = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial cell state
        
        self.h_r = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial hidden state
        self.c_r = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial cell state

        self.h_f2l = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial cell state
        self.c_f2l = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial cell state

        self.h_f2r = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial hidden state
        self.c_f2r = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial hidden state

        self.h_h2h = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial hidden state
        self.c_h2h = torch.zeros([self.num_layers, 1, self.hidden_size]) # Initial cell state


    def prepare_input(self, x: torch.tensor):
        x = x.float()
        x = x.unsqueeze(1)

        # Input dims: N x L x IN
        source_body = x[0][0][0].clone()
        source_left = x[0][0][100].clone()
        source_right = x[0][0][121].clone()

        body = x[0][0][:17]
        left = x[0][0][91:112]
        right = x[0][0][112:]

        w_left = torch.max(left[:, 0]) - torch.min(left[:, 0])
        h_left = torch.max(left[:, 1]) - torch.min(left[:, 1])
        w_right = torch.max(right[:, 0]) - torch.min(right[:, 0])
        h_right = torch.max(right[:, 1]) - torch.min(right[:, 1])
        w_body = body[5][0] - body[6][0]
        h_body = 4 * w_body

        face2left = left - source_body
        face2right = right - source_body
        hand2hand = left - right
        left -= source_left
        right -= source_right

        if w_left != 0. and h_left != 0:
            left[:, 0] /= w_left
            left[:, 1] /= h_left

        if w_right != 0. and h_right != 0:
            right[:, 0] /= w_right
            right[:, 1] /= h_right

        if w_body != 0. and h_body != 0:
            face2left[:, 0] /= w_body
            face2right[:, 1] /= h_body
            hand2hand[:, 0] /= w_body
            hand2hand[:, 1] /= h_body

        x = torch.concat([left, right, face2left, face2right, hand2hand])
        x = x.view(1, 1, (21 + 21 + 21 + 21 + 21) * 2)
        
        return x
    

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.prepare_input(x)
        x = torch.permute(x, (1, 0, 2))

        l, (self.h_l, self.c_l) = self.l(x[:, :, :21*2], (self.h_l, self.c_l))
        r, (self.h_r, self.c_r) = self.r(x[:, :, 21*2:42*2], (self.h_r, self.c_r))
        f2l, (self.h_f2l, self.c_f2l) = self.f2l(x[:, :, 42*2:63*2], (self.h_f2l, self.c_f2l))
        f2r, (self.h_f2r, self.c_f2r) = self.f2r(x[:, :, 63*2:84*2], (self.h_f2r, self.c_f2r))
        h2h, (self.h_h2h, self.c_h2h) = self.h2h(x[:, :, 84*2:], (self.h_h2h, self.c_h2h))

        l = l[-1, :, :]
        r = r[-1, :, :]
        f2l = f2l[-1, :, :]
        f2r = f2r[-1, :, :]
        h2h = h2h[-1, :, :]

        out = torch.concat([l, r, f2l, f2r, h2h], dim=1)
        out = self.fuse(out)
        
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
        self.h = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial hidden state
        self.c = torch.zeros([self.num_layers, 1, self.hidden_size])  # Initial cell state

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