import torch
import torch.nn as nn
import math
import numpy as np

class Positional_Encoding(nn.Module):
    def __init__(self, model_size):
        super(Positional_Encoding, self).__init__()
        # model_size is the embedding size of each embedded token
        self.model_size = model_size
    
    def forward(self, seq_len, embedding_size):
        '''
        Here is the implementation of positional encoding
        seq_len stands for the input.size[1]
        embedding_size stands for the model_size shown previously
        '''
        positional_encoding = np.zeros((seq_len, embedding_size))
        # token stands for the position in the sentence
        for token in range(seq_len):
            # pos stands for the position in the embedding sequence
            for pos in range(embedding_size):
                if pos % 2 == 0:
                    positional_encoding[token][pos] = math.sin(token/10000 ** (2 * pos/self.model_size))
                else:
                    positional_encoding[token][pos] = math.cos(token/10000 ** (2 * pos/self.model_size))

        return torch.from_numpy(positional_encoding)