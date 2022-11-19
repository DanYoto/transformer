import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from Embedding import Embedding
import Config

class Tranformer_layer(nn.Module):
    def __init__(self):
        super(Tranformer_layer, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    
    def forward(self, x):
        x_encoder, x_decoder= x
        encoder_out = self.enc(x_encoder)
        decoder_out = self.dec(x_decoder, encoder_out)
        return (encoder_out, decoder_out)
    
class Transformer(nn.Module):
    def __init__(self, N, vocab_size, output_dim):
        super(Transformer, self).__init__()
        self.embedding_encoder = Embedding(vocab_size, Config.embed_size)
        self.embedding_decoder = Embedding(vocab_size, Config.embed_size)

        self.output_dim = output_dim
        self.linear = nn.Linear(Config.embed_size, output_dim)
        self.softmax = nn.Softmax()
        self.model = nn.Sequential(*[Tranformer_layer() for i in range(N)])
    
    def forward(self, x):
        x_input, x_output = x
        x_input = self.embedding_decoder(x_input)
        x_output = self.embedding_decoder(x_output)

        _, output = self.model((x_input, x_output))

        output = self.linear(output)
        output = self.softmax(output)

        return output
