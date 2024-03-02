#################################
# Name: BP.py
# Function: Backpropagation network from built - settings - predict.
# Author: Copied from Morvan_Zhou's website: https://yulizi123.github.io/tutorials/machine-learning/torch/3-01-regression/
# Date: 02/03/2024
# Environment:  Python - 3.11.5
#               Pytorch - 2.2.0
#################################
import torch
from torch import nn
import torch.nn.functional as Func


class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()

        self.hidden = torch.nn.Linear(1, 32)
        self.predict = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = Func.relu(self.hidden(x))
        x = self.predict(x)
        return x


bp_net = BP()
print(bp_net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = nn.MSELoss()

for step in range(50):
    x = torch.from_numpy(train_set_x[np.newaxis, :, np.newaxis]).float()
    y = torch.from_numpy(train_set_y[np.newaxis, :, np.newaxis]).float()

    prediction = bp_net(x)

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.data)
