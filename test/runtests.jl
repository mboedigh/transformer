import Transformers
lre(x) = -log10.(abs.(x))

d_strlen = 20;   # maximum sequence length
d_vocab = 11;
d_model = 512;
n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
n_layers = 6;
P_DROP = 0.1;    # turn to 0.0 for testing otherwise it is not deterministic

model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP);

function data_gen(batch_size, d_vocab, d_strlen, d_model, nbatches)
    # "Generate random data for a src-tgt copy task. i.e. the target is an exact copy of the source"
    data = [ rand(1:d_vocab, batch_size, d_strlen ) for in in 1:nbatches]
    src = data;
    tgt = copy(data);
    src_mask = [ones( 1, d_model) for i in 1:nbatches]
    tgt_mask = [tril( ones(d_model,d_model)) for i in 1:nbatches]
    (src, tgt, src_mask, tgt_mask)
end

batch_size = 30;
n_batches = 2;
(src, tgt, src_mask, tgt_mask) = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);
batch = src[1];
datum = batch[1,:]; 

# testing (bottom up)
# each element of datum is a input in a sequence. Embedding is a d_model vector describing each element 
input = datum |> model.source_embedding;

# LayerNorm
# implemented by Flux, but Flux normalizes columns and I need to normalize rows (or transpose everything), so that each example (input row) is normalized
l = LayerNorm( d_model);
input = rand(d_strlen, d_model)
x = l(input,2);
@assert all( lre(mean(x,dims=2)) .> 6  )
@assert all( lre(std(x,dims=2) .- 1) .> 6  )
Flux.params(l); # I don't know what to test here, but at least it returns something reasonable

# Dropout
# Dropout inactive
d = Dropout(0); # p_drop = 0, so dropout is inactive 
x = rand(100,100);
@assert x == d(x)

# dropout p_drop = 0.1
d = Dropout(0.1);
xo = d(x);
i = xo .!= 0;
@assert xo[i] == x[i]

# dropout test mode (scale by 1-p)
d.training = false;
xo = d(x);
@assert xo == x*(1-d.p)

# sublayer
# adds the input to the layer-normalized output of a function. 
# test normalization
f(x) = 2x;
x = [1.0 3 5; 2 4 6];
s = Sublayer(f, size(x,2))
out = s(x; p_drop = 0 );
@assert all( lre(out - [-1 0 1.; -1 0 1]) .> 10 )
ps = Flux.params(s); # I don't know what to test here, but at least it returns something reasonable

# test function and normalization
x = rand(2,4)
xx = f(x)+x
varxx = var(xx, dims=2) # 9va
s = Sublayer( f, size(x,2))
out = s(x,p_drop=0)
@assert all( lre(out - (xx .- mean(xx,dims=2))./sqrt.(varxx)) .> 10 ) # accurate to at least 10 digits

# RepeatedLayers
# Simple chain of identical layers. Output from one layer is passed to the next. 
r = RepeatedLayer(f, 6)
x = [1.0 3 5; 2 4 6];
out = r(x);
@assert all( lre(out -  x * 2^6 ) .> 10) # accurate to at least 10 digits

r = RepeatedLayer(f,0) # 0 layers returns x (runs function 0 times)
@assert x == r(x)

r = RepeatedLayer(f,1) # runs the function 1 time
@assert all( lre(r(x) -  x * 2 ) .> 10) # accurate to at least 10 digits

r = RepeatedLayer(s, 3);
ps = Flux.params(r); # I don't know what to test here, but at least it returns something reasonable

# Embedding
# how to test word embedding?????
W = Flux.param(rand(Float32, d_vocab, d_model)); # need to create weights outside Embedding,to share them between embedding layers and pre-softmax transform
embedding           = Embedding(W);   
x = datum |> embedding; 
ps = Flux.params( embedding );
@assert length(ps) == 1

ch = Flux.Chain( embedding, embedding);
ps = Flux.params(ch)
@assert length(ps) == 1 # W is used only once, no duplicates

# PositionalEncoding
pe = PositionalEncoding(100, d_model, 0);
z = zeros( 100, d_model) |> pe;
# plot( 1:100, z[:,5:8]); # takes too long to load Plots and dra!

ff = PositionwiseFeedForward(d_model,d_model*4,d_model);
input = randn(2,d_model);
output = ff(input);
o1 = ff.w1(input');
o2 = ff.w2(o1)';
@assert all( lre(o2 -  output ) .> 10) # accurate to at least 10 digits
ps = Flux.params(ff); # I don't know what to test here, but at least it returns something reasonable
@assert(length(ps)== 4 )

# Attention
d_attn = Int32(d_model/n_heads);
a = Attention( d_model,  d_attn );
x = randn(10,d_model)
Q,K,V = x*a.Q, x*a.K, x*a.V;
score = ((Q*K')/sqrt(d_attn))*V;
@assert all( lre( a(x,x,x) - Flux.softmax(score')') .> 10)
ps = Flux.params(a)
@assert(length(ps)== 3 )

# MultiHeadedAttention
mha                 = MultiHeadedAttention( n_heads, d_model, d_attn);
ps = Flux.params(mha); 
@assert(length(ps)== n_heads*3 + 1)

# Encoder Layer
encoder = Encoder( mha, ff, p_drop = P_DROP);
ps = Flux.params(encoder)
 # 4 params from ff, 25 from mha (24 from heads, and one final projection) and 4 from Sublayer
@assert length(ps) == 4 + n_heads*3 + 1 + 2*2

# Encoder Stack (Unit with sublayers wrapping MH self-attention -> sublayer wrapping PosFF )
encoder_stack       = RepeatedLayer( Encoder(mha, ff, p_drop = P_DROP), n_layers);
ps = Flux.params(encoder_stack);
@assert length(ps) == 6*33 

c = deepcopy;
# Decoder (Unit with sublayers wrapping MH self-Attenion -> sublayer wrapping MH encoder_decoder-attention -> sublayer wrapping PosFF)
decoder = Decoder( c(mha), c(mha), c(ff), p_drop = P_DROP);
ps = Flux.params(decoder);
# 4 params from ff, 25 from mha and 6 from 3 Sublayers
@assert length(ps) == 4 + n_heads*3 + (n_heads*3 + 2) + 3*2

# The decoder stack
decoder_stack       = RepeatedLayer( decoder, n_layers);
ps = Flux.params(decoder_stack);
@assert length(ps) == 6*60 

# Transformer model
ps = Flux.params( model);
# one for two embedding layers and one generation layer (shared) + 6*encoder_stack + 6*decoder_stack + 2 output (gain + bias)
@assert length(ps) == 1 + 6*33  + 6*60 + 2
