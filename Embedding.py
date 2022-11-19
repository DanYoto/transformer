#In this file, it mainly contains context embedding and positional embedding
import torch
import torch.nn as nn
import Config
import numpy as np
import random

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        # First, for each sentence, we need to pad them into same length
        for i in range(len(x)):
            # If the length is smaller, it should be padded with UNK
            if len(x[i]) < Config.padding_size:
                x[i].extend([Config.UNK] * (Config.padding_size - len(x[i])))
            # If the length is larger, it should be truncated
            else:
                x[i] = x[i][:Config.padding_size]
        
        x = self.embedding(torch.tensor(x))
        return x

if __name__ == '__main__':
    # input should be list type
    in_list = []
    for i in range(64):
        tmp_in = random.sample(range(0, 100), 20)
        in_list.append(tmp_in)

    emb = Embedding(600, 60)
    tmp_emb = emb(in_list)
    print(tmp_emb.size())