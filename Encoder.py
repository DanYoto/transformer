import torch
import torch.nn as nn
from Positional_Encoding import Positional_Encoding
from Feed_Forward import Feed_Forward
from Multihead_Attention import Multihead_Attention
from Add_norm import Add_norm
import Config

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding(Config.embed_size)
        self.multihead_atten = Multihead_Attention(Config.embed_size, Config.dim_v, Config.dim_k, Config.n_heads)
        self.feed_forward = Feed_Forward(Config.embed_size, hidden_dim = 1024)
        self.add_norm = Add_norm()
    
    def forward(self, x):
        x += self.positional_encoding(x.shape[1], Config.embed_size)
        output = self.add_norm(x, self.multihead_atten, y = x)
        output = self.add_norm(output, self.feed_forward)

        return output

if __name__ == '__main__':
    tmp_in = torch.rand(64, 30, 60)
    enc = Encoder()
    tmp_out = enc(tmp_in)
    print(tmp_out.size())