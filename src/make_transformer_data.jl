using Statistics
using LinearAlgebra
import Flux
using Transformers   
##
d_strlen = 10;   # maximum sequence length
d_vocab = 11;
d_model = 512;
n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
n_layers = 6;    # 6 in the paper
P_DROP = 0.1f0;    

model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = n_layers, n_heads = n_heads);

n_batches = 20;
batch_size = 30;
dataset = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);
batch = dataset[1];  # for debugging at REPL
datum = batch[1,:];

model(datum,datum)

