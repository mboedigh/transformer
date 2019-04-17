using  Statistics
using  LinearAlgebra
import Flux
using  Transformers   

function data_gen(batch_size, d_vocab, d_strlen, d_model, nbatches)
    # "Generate random data for a src-tgt copy task. i.e. the target is an exact copy of the source"
    data = [ rand(1:d_vocab, batch_size, d_strlen ) for in in 1:nbatches]
    # tgt = copy(data);
    # src_mask = [ones( 1, d_model) for i in 1:nbatches]
    # tgt_mask = [tril( ones(d_model,d_model)) for i in 1:nbatches]

    for i = 1:size(data,1)
       d = view( data[i], :, 1 ); 
       fill!(d, 1);
    end
    return data
end

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

