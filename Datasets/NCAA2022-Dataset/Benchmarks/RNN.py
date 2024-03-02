#################################
# Name: RNN.py
# Function: Recurrent neural network from built - settings - predict.
# Author: Copied from Morvan_Zhou's website: https://yulizi123.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/
# Date: 02/03/2024
# Environment:  Python - 3.11.5
#               Pytorch - 2.2.0
#################################
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
loss_func = nn.MSELoss()

for step in range(50):
    x = torch.from_numpy(train_set_x[np.newaxis, :, np.newaxis]).float()
    y = torch.from_numpy(train_set_y[np.newaxis, :, np.newaxis]).float()

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.data)
