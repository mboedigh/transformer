using Transformers
lre(x) = -log10.(abs.(x))

d_strlen = 10;   # maximum sequence length
d_vocab = 11;
d_model = 512;
n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
n_layers = 6;
P_DROP = 0.1;    # turn to 0.0 for testing otherwise it is not deterministic

model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP);

batch_size = 30;
n_batches = 2;
src = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);
batch = src[1];
datum = batch[1,:]; 

# testing (bottom up)
# each element of datum is a input in a sequence. Embedding is a d_model vector describing each element 
input = datum |> model.source_embedding;

# LayerNorm
# implemented by Flux, but Flux normalizes columns and I need to normalize rows (or transpose everything), so that each example (input row) is normalized
l = Transformers.LayerNorm( d_model);
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

# dropout p_drop = 0.1, scale = 1/(1-p_drop)
d = Dropout(0.1);
xo = d(x);
i = xo .!= 0;
@assert xo[i] == x[i]*(1/(1-d.p))

# sublayer
# adds the input to the layer-normalized output of a function. 
# test normalization
f(x) = 2x;
x = [1.0 3 5; 2 4 6];
s = Sublayer(f, size(x,2), 0.0)
out = s(x);
@assert all( lre(out - [-1 0 1.; -1 0 1]) .> 10 )
ps = Flux.params(s); # I don't know what to test here, but at least it returns something reasonable

# test function and normalization
x = rand(2,4)
xx = f(x)+x
varxx = var(xx, dims=2) # 9va
s = Sublayer( f, size(x,2), 0.0)
out = s(x);
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
# each word in a sequence is a different point along a sinusoid. AND, each feature in d_model is a different family of sinusoid (frequency and sine versus cosine)
pe = PositionalEncoding(100, d_model, 0);
z = zeros( 100, d_model) |> pe;
# plot( 1:100, z[:,5:8]); # takes too long to load Plots and draw!

ff = PositionwiseFeedForward(d_model,d_model*4,d_model);
input = randn(2,d_model);
output = ff(input);
o1 = ff.w1(input);
o2 = ff.w2(o1);
@assert all( lre(o2 -  output ) .> 10) # accurate to at least 10 digits
ps = Flux.params(ff); # I don't know what to test here, but at least it returns something reasonable
@assert(length(ps)== 4 ) # 2 layers each with gain and bias

# Attention 
d_attn = Int32(d_model/n_heads);
x      = randn(10,d_model)
out = mha(x,x,x)
range = 1:d_attn
Q,K,V = x*view(mha.Q.W,:,range), x*view(mha.K.W,:,range), x*view(mha.V.W, :,range);
b     = view(mha.Q.b,range) # defaults to 0s so we can ignore
score = ((Q*K')/sqrt(d_attn));
sm_score     = Flux.softmax( score' )';
Z = sm_score*V;

@assert all( lre( a(x,x,x) - score) .> 10)

# MultiHeadedAttention
mha                 = MultiHeadedAttention( n_heads, d_model, d_attn);
ps = Flux.params(mha); 
# 2 params for Q,K,V and 2 for final projection
@assert(length(ps)== 6 + 2)
mha_params = length(ps)

# Encoder Layer
encoder = model.encoder_stack[1];
ps = Flux.params(encoder);
encoder_params = length(ps)
 # 4 params from ff, n_layers*mha (48), 2 from LayerNorm in each of 2 Sublayers
@assert encoder_params == 4 + mha_params + 2*2

# Encoder Stack (Unit with sublayers wrapping MH self-attention -> sublayer wrapping PosFF )
encoder_stack   = model.encoder_stack;
encoder_stack_params = length(Flux.params(encoder_stack))
@assert encoder_stack_params == 6*encoder_params

# Decoder (Unit with sublayers wrapping MH self-Attenion -> sublayer wrapping MH encoder_decoder-attention -> sublayer wrapping PosFF)
decoder = model.decoder_stack[1];
decoder_params = length(Flux.params(decoder));
# 4 params from ff,  6*from self attention and 6 from src attention + two final linear from each mha and and from 3 Sublayers with 2 each
@assert length(ps) == 4 + 6 + 6 + 2*2 + 3*2

# The decoder stack
decoder_stack = model.decoder_stack;
decoder_stack_params = length(Flux.params(decoder_stack));
@assert decoder_stack_params == n_layers*decoder_params

# Transformer model
ps = Flux.params( model);
# two separate embedding layers (currently, but this is not to spec) and one generation layer (shared) + 6*encoder_stack + 6*decoder_stack + 2 output (gain + bias)
@assert length(ps) == 2 + encoder_stack_params  + decoder_stack_params+ 2
