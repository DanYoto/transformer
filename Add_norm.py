import torch
import torch.nn as nn
import Config

class Add_norm(nn.Module):
    def __init__(self):
        super(Add_norm, self).__init__()
        self.dropout = nn.Dropout(Config.p)

    def forward(self, x, sub_layer, **kwargs):
        sub_output = self.dropout(sub_layer(x, **kwargs))
        x = x + sub_output

        layer_norm = nn.LayerNorm(x.size()[1:])
        output = layer_norm(x)
        return output