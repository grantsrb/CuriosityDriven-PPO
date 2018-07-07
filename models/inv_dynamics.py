import torch.nn as nn

class InvDynamics(nn.Module):
    def __init__(self, h_size, action_size):
        super(InvDynamics, self).__init__()
        self.h_size = h_size
        self.action_size = action_size

        self.inv_dyn = nn.Sequential(nn.Linear(2*h_size, h_size), nn.ReLU(),
                                    nn.Linear(h_size, h_size), nn.ReLU(),
                                    nn.Linear(h_size, action_size))

    def forward(self, x):
        return self.inv_dyn(x)

