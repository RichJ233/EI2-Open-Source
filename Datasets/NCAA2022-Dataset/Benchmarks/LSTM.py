#################################
# Name: LSTM.py
# Function: Long short term network from built - settings - predict.
# Author: Rich inspired by Morvan_Zhou's website: https://yulizi123.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/
# Date: 02/03/2024
# Environment:  Python - 3.11.5
#               Pytorch - 2.2.0
#################################
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, (h_n, h_c) = self.rnn(x, h_state)

        r_out = rout.view(-1, 32)
        outs = self.out(r_out)
        return outs.view(-1, outs.size(0), outs.size(1)), (h_n, h_c)

lstm = LSTM()
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=0.02)
loss_func = nn.MSELoss()
h_state = None

for step in range(50):
    x = torch.from_numpy(train_set_x[np.newaxis, :, np.newaxis]).float()
    y = torch.from_numpy(train_set_y[np.newaxis, :, np.newaxis]).float()

    prediction, (h_0, h_1) = lstm(x, h_state)

    h_0 = h_0.data
    h_1 = h_1.data
    h_state = (h_0, h_1)

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.data)
