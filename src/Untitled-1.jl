d_strlen = 10;   # maximum sequence length
d_vocab = 11;
d_model = 512;
n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
n_layers = 6;
P_DROP = 0.1;    # turn to 0.0 for testing otherwise it is not deterministic

n_batches = 20;
batch_size = 30;
batches = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);

batch = batches[1];  # for debugging at REPL
datum = batch[1,:];
target = datum[1:end-1];
target_y = datum[2:end];


model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP);
e = model.target_embedding(datum);
d = model.decoder_stack[1];
o = d(e,e);
yhat = model.generator(e);

l = loss( yhat, datum, d_vocab);
Flux.back!(l);
ps = Flux.params(model);
Flux.Optimise.update!(opt, ps);
model.target_embedding.W