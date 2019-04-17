
include("src/transformer_demo.jl");
include("src/make_transformer_data.jl");
include("src/Softmax.jl");

x = encode(model,datum);
q,k,v = (x,x,x)

mha = model.encoder_stack[1].mha.fn;
Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
n_k   = (size(q,2)//mha.n_heads).num;
h = 1;
query  = [view(Q, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
key    = [view(K, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
value  = [view(V, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];

# These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
scale = typeof(q.data[1])(1.0/sqrt(n_k))
# this works with some (all?) versions of Flux
o = [attention(z[1],z[2], z[3],scale) for z in zip(query,key,value)];
o = hcat(o...); # supposedly slower than reduce(hcat,o), but produces different outputs

# this fails with some versions of Flux, due to unsupported softmax! with array of tracked real
o = [attention(z[1],z[2], z[3],scale) for z in zip(query,key,value)];
o = reduce( hcat, o); # avoids splat operator (o = hcat(o...)), which is supposedly slower

using Flux
tracked_array = Flux.param(Flux.glorot_uniform(10,3))
Flux.softmax(tracked_array) # all good

tracked_reals = reduce(hcat, [tracked_array,tracked_array]);
Flux.softmax(tracked_reals) # fails with MethodError: no method matching softmax!(...)

