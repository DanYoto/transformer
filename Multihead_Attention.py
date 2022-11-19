# This is to implement the scaled Dot-Product Attention
import torch
import torch.nn as nn
import math
import numpy as np

class Multihead_Attention(nn.Module):
    def __init__(self, embed_size, dim_v, dim_k, n_heads):
        super(Multihead_Attention, self).__init__()
        # The size of Query and Key should be the same, which is set to dim_k
        # The size of value is set to dim_v
        # As we are using multihead attention, the dimension should be multiple times of projected size
        self.embed_size = embed_size
        #dim_k and dim_v is the total size, which indicates that dim_k and dim_v should be multiple times of n_heads
        # The benenfits are that there is no need for concatenation any more
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.get_query = nn.Linear(embed_size, dim_k)
        self.get_key = nn.Linear(embed_size, dim_k)
        self.get_value = nn.Linear(embed_size, dim_v)

        self.output = nn.Linear(dim_v, embed_size)
        self.norm_factor = 1/math.sqrt(dim_k // self.n_heads) # in order to get int instead of float
        self.softmax = nn.Softmax()

    def generate_mask(self, dim):
        '''
        Here is designed for the masked multi-head attention
        '''
        # This is designed to prevent decoder could obtain the future information
        initial_matrix = np.ones((dim, dim))
        # Here is to get the lower triangular array
        mask = torch.tensor(np.tril(initial_matrix))
        return mask
    
    def forward(self, x, y, require_mask = False):
        '''
        attention for encoder, x and y should be the same
        the size is [batch_size, sequence_len, embedding_size]

        attention for decoder, x and y are different
        x should be the output of previous decoder layer -- query
        y shoud be the output of encoder -- keys and values
        '''

        assert (self.dim_k % self.n_heads == 0), 'dim_k should be divided by n_heads' 
        assert (self.dim_v % self.n_heads == 0), 'dim_v should be divided by n_heads'
        Querys = self.get_query(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        Keys = self.get_key(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        Values = self.get_value(y).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.n_heads)
        attention_score = torch.matmul(Querys, Keys.permute(0, 1, 3, 2))

        if require_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score = attention_score.masked_fill(mask == 0, value = 1e-9)

        out = torch.matmul(self.softmax(attention_score), Values).reshape(y.shape[0], y.shape[1], -1)

        res = self.output(out)
        return res


if __name__ == '__main__':
    input_tmp = torch.rand((64, 30, 20))
    multi_atten = Multihead_Attention(20, 200, 150, 10)
    out_tmp = multi_atten(input_tmp, input_tmp, True)
    print(out_tmp.size())
    # output size should be the same as input size