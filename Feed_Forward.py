import torch
import torch.nn as nn

class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # in the paper, input_dim = 512, hidden_dim = 2048
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        '''
        Here is the implementation for Position-wise Feed-Forward Networks
        '''
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output