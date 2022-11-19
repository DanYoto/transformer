# vocab_size stands for the total size of vocabulary + UNK
vocab_size = 600
# embed_size stands for the size for each embedded token
embed_size = 60

# This depends how many heads you wanna add for multihead attention
n_heads = 10


dim_k  = 600
dim_v = 600

assert (dim_k % n_heads == 0), 'dim_k should be divided by n_heads'
assert (dim_v % n_heads == 0), 'dim_v should be divided by n_heads'
#dim_k and dim_v stands for the total dimension of keys and values

# Here is the length of each embedded sentence, if it is smaller/larger than the padding_size, 
# it should be padded/trnuncated into padding_size
padding_size = 30
#UNK is for the word used for oov(out of vocabulary)
UNK = 599

N = 6
# probability for dropout
p = 0.1