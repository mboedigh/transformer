using LinearAlgebra
using Flux
using Transformers
using Statistics
using Test: @test, @test_broken, @testset

@testset "Transformer Tests" begin

lre(x) = -log10.(abs.(x))

d_model = 512;
d_strlen = 64;
d_vocab = 13;
n_heads = 8;
model = Transformer( ;d_model = d_model, max_seqlen = d_strlen, d_vocab=d_vocab, n_heads = 8);

datum = [1 4 5 6 7 8 9 10 2];

# each element of datum is a single input sequence. Embedding is a d_model vector describing each element 
input = datum |> model.source_embedding;  # this only needs to run without throwing an error

# LayerNorm
# implemented by Flux, but Flux normalizes columns and I need to normalize rows (or transpose everything), so that each example (input row) is normalized
l = Transformers.LayerNorm( d_model);
input = rand(d_strlen, d_model)
x = l(input,2);
@test all( lre(mean(x,dims=2)) .> 6  )    # good to 6 digits
@test all( lre(std(x,dims=2,corrected=false) .- 1) .> 5 ) # good to 5 digits

# Dropout
# Dropout inactive
d = Transformers.Dropout(0.0f0); # p_drop = 0, so dropout is inactive 
x = rand(100,100);
@test x == d(x)

# dropout p_drop = 0.1, scale = 1/(1-p_drop)
d = Transformers.Dropout(0.1);
xo = d(x);
i = xo .!= 0;
@test all( lre( xo[i] - x[i]*(1/(1-d.p))) .> 6)

# sublayer
# adds the input to the layer-normalized output of a function. 
# test normalization
f(x) = 2x;
x = [1.0 3 5; 2 4 6];
s = Sublayer(f, size(x,2), p_drop = 0.0f0)
out = s(x);
l = Transformers.LayerNorm( 3)(x)
@test  all( lre(out - l) .> 10 ) 

# RepeatedLayers
# Simple chain of identical layers. Output from one layer is passed to the next. 
ds = [(x)->2x for i = 1:6];
r = RepeatedLayer(ds)
x = [1.0 3 5; 2 4 6];
out = r(x);
@test all( lre(out -  x * 2^6 ) .> 10) # accurate to at least 10 digits

# test multiple arguments
ds = [(x,b)->2x .+ b for _ = 1:6];
r = RepeatedLayer(ds)
x = [1.0 3 5; 2 4 6];
f(x) = 2x .+ 1;
t = f(f(f(f(f(f(x))))));
out = r(x, 1);
@test all( lre(out -  t) .> 10) # accurate to at least 10 digits

ds = [(x)->2x for i = 1:0];
r = RepeatedLayer(ds) 
@test x == r(x)

ds = [(x)->2x for i = 1:1];
r = RepeatedLayer(ds) 
@test all( lre(r(x) -  x * 2 ) .> 10) # accurate to at least 10 digits

# Embedding
# I don't know how to test word embedding, except that it runs
W = Flux.param(rand(Float32, d_vocab, d_model)); # need to create weights outside Embedding,to share them between embedding layers and pre-softmax transform
embedding           = Embedding(W);   
x = datum |> embedding; 
ps = Flux.params( embedding );
@test length(ps) == 1 # only the Look up table of parameters. each word (token) in the sequence has a d_model vector of parameters

# PositionalEncoding
# each word in a sequence is a different point along a sinusoid. AND, each feature in d_model is a different family of sinusoid (frequency and sine versus cosine)
pe = PositionalEncoding(100, d_model, 0);
z = zeros( 100, d_model) |> pe;
# first feature is a sine
# plot( 1:100, z[:,5:8]); # takes too long to load Plots and draw!

ff = PositionwiseFeedForward(d_model,d_model*4,d_model);
input = randn(2,d_model);
output = ff(input);
o1 = ff.w1(input);
o2 = ff.w2(o1);
@test all( lre(o2 -  output ) .> 10) # accurate to at least 10 digits
ps = Flux.params(ff); # I don't know what to test here, but at least it returns something reasonable
@test length(ps)== 4  # 2 layers each with gain and bias

# Attention 
d_attn = Int32(d_model/n_heads);
x      = randn(Float32, 10,d_model)
range = 1:d_attn
mha                 = MultiHeadedAttention( n_heads, d_model, d_attn);

Q,K,V = x*view(mha.Q.W,:,range), x*view(mha.K.W,:,range), x*view(mha.V.W, :,range);
scale = Float32(1/sqrt(d_attn))
score = (Q*K')*scale;
sm_score     = Flux.softmax( score' )';
Z = sm_score*V;
@test all( lre( attention(Q,K,V,nothing,scale) - Z) .> 10)

# MultiHeadedAttention
mha                 = MultiHeadedAttention( n_heads, d_model, d_attn);
ps = Flux.params(mha); 
# 2 params for Q,K,V and 2 for final projection
@test(length(ps)== 6 + 2)
mha_params = length(ps)

# Encoder Layer
encoder = model.encoder_stack[1];
ps = Flux.params(encoder);
encoder_params = length(ps)
# 4 params from ff, 8 params for Q,K,V,W in mha, 2 from LayerNorm in each of 2 Sublayers
@test encoder_params == 4 + mha_params + 2*2 

# Encoder Stack (Unit with sublayers wrapping MH self-attention -> sublayer wrapping PosFF )
encoder_stack   = model.encoder_stack;
encoder_stack_params = length(Flux.params(encoder_stack))
n_layers = length(encoder_stack.layers );
@test encoder_stack_params == n_layers*encoder_params

# Decoder (Unit with sublayers wrapping MH self-Attenion -> sublayer wrapping MH encoder_decoder-attention -> sublayer wrapping PosFF)
decoder = model.decoder_stack[1];
decoder_params = length(Flux.params(decoder));
# 4 params from ff,  8 from self attention and 8 from src attention + two final linear from each mha and and from 3 Sublayers with 2 each
@test decoder_params == 4 + 2*mha_params + 3*2  

# The decoder stack
decoder_stack = model.decoder_stack;
decoder_stack_params = length(Flux.params(decoder_stack));
@test decoder_stack_params == n_layers*decoder_params

# Transformer model
ps = Flux.params( model);
# one shared embedding layers and one generation layer (gain and bias) + encoder_stack + decoder_stack
@test length(ps) == 1 + 2 + encoder_stack_params  + decoder_stack_params

# test batch embedding is the same as sequence embedding (does not demonstrate correctness! only demonstrates consistency)
source = [1   6  13  11   6  7  12  12   7  2
1   6   8  12   6  6  10  13   7  2
1   8   7  13   5  5   5   6  13  2
1  12   9   9  11  7   9   7   9  2
1  11  10   6   5  6   6   6  11  2];
slen = size(source,2)
setdropoutmode!(model, false)
embedded_batch = embed(model,source);
for k in 1:size(source,1)
   embedded_seq  = embed(model,source[k,:]);
   @test isequal( embedded_batch[ (1:slen) .+ slen*(k-1),:], embedded_seq)
end

# test batch encoding 
slen = size(source,2)
encoded_batch = encode( model, source );
test_seqs = rand( 1:size(source,1), 3);
for k in test_seqs
   encoded_seq   = encode( model, source[k,:]);
   @test isequal( encoded_batch[ (1:slen) .+ slen*(k-1),:], encoded_seq)
end


########################################
# compare outputs from single and batch mode MHA
target = vec([ 1   6   7   7   8   5   5  11   6   9   8   8   9   2   3   3  3 ]);
# initialize parameters
d_model = 64;
mha = MultiHeadedAttention( 4, d_model, (d_model//4).num; init = Flux.glorot_uniform);
tmask= getmask(target);
# fit model
x    = randn( Float32, length(target), d_model); # random input
batch_mode  = mha(x,x,x,1, tmask); # batch mode with 1 sequence
seq_mode    = mha(x,x,x,tmask); # seq mode
@test norm( batch_mode.data - seq_mode.data) ≈ 0


x2    = [x;x]
batch_mode  = mha(x2,x2,x2,2, [tmask;tmask]); # batch mode with 2 sequences
batch_mode  = batch_mode[ length(target)+1:end,:] 
@test norm( batch_mode.data - seq_mode.data) ≈ 0

#######################################
# make sure transformer does not drop gradients (all parameters are actualy updated)
setdropoutmode!(model,false);
stepnum = 1;
warmup = 400;
opt = Flux.ADAM( 1, (0.9, 0.98) )
d_model = 512;
d_strlen = 64;
d_vocab = 13;
n_heads = 8;
model = Transformer( ;d_model = d_model, max_seqlen = d_strlen, d_vocab=d_vocab, n_heads = 8); # new model and parameters
ps = Flux.params(model);
ps0 = deepcopy( ps );
lbar = transformer_loss(model, source, source );  # call loss function to save the result
gs = Flux.gradient( ()->lbar,ps);  
Flux.Optimise.update!(opt, ps, gs);
d = [ sum(p[1]).data - sum(p[2].data) for p in zip(ps, ps0)]
k = d .== 0;
@test !any(k)  # this is somewhat random because some parameters are being updated but by so little that they appear to be 0 (i.e. if run again on new data it may pass, and so is being updated)

end