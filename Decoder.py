import torch
import torch.nn as nn
from Positional_Encoding import Positional_Encoding
from Feed_Forward import Feed_Forward
from Multihead_Attention import Multihead_Attention
from Add_norm import Add_norm
import Config

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = Positional_Encoding(Config.embed_size)
        self.multihead_atten = Multihead_Attention(Config.embed_size, Config.dim_v, Config.dim_k, Config.n_heads)
        self.feed_forward = Feed_Forward(Config.embed_size, hidden_dim = 1024)
        self.add_norm = Add_norm()
    
    def forwards(self, x, encoder_output):
        x += self.positional_encoding(x.shape[1], Config.embed_size)
        output = self.add_norm(output, self.multihead_atten, y = x, require_mask = True)
        output = self.add_norm(output, self.multihead_atten, y = encoder_output, require_mask = True)
        output = self.add_norm(output, self.feed_forward)
        return output