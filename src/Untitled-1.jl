

# turn off dropout so I can look at whether parameters are changing
setdropoutmode(model, false)
ps = Flux.params(model);
ps1 = deepcopy(ps);
first(ps1)

opt.eta = learn_rate(400, 400)
my_loss(batch, target, target_y) = transformer_batch(model, batch, target, target_y);

target = view(batch, :, 1:size(batch,2)-1);
target_y = view(batch, :, 2:size(batch,2));
data = [(batch, target, target_y)];
Flux.train!(my_loss, ps, data, opt)


deltas = [norm(p1.data -p2.data) for (p1, p2) in zip(ps1, ps)];
   
##

d_strlen = 10;   # maximum sequence length
d_vocab = 11;
d_model = 512;
n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
n_layers = 6;    # 6 in the paper
P_DROP = 0.1;    

model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = n_layers, n_heads = n_heads);

n_batches = 20;
batch_size = 30;
dataset = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);
batch = dataset[1];  # for debugging at REPL
datum = batch[1,:];

model(datum,datum)

