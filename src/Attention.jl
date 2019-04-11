
import Flux

function attention( q, k, v, scale)
    score = scale.*( q*k');
    Flux.softmax(score')'*v  
end

function attention( q, k, v, scale, mask1, mask2)
    score = (scale.*( q*k')).*mask1 + mask2;
    Flux.softmax(score')'*v  
end

# Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, 
# keys and values h times with different, learned linear projections to dq, dk and dv dimensions, respectively. 
# I am doing these all in one matrix multiply
struct MultiHeadedAttention
    n_heads
    Q
    K
    V

    # On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
    # These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
    W;
end

function MultiHeadedAttention( n_heads::Integer, d_in::Integer, d_k::Integer; init = Flux.glorot_uniform) 
    return MultiHeadedAttention( n_heads, [Linear(d_in, d_k*n_heads, initW=init) for i in 1:4]...);
end

# On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
# These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
# The online version of this (Annotated Transformer uses a gain = bias)
function (mha::MultiHeadedAttention)(q,k,v)

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_k   = (size(q,2)//mha.n_heads).num;
    h = 1;
    query  = [view(Q, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    key    = [view(K, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    value  = [view(V, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];

    o = hcat( [attention(z[1],z[2], z[3],1.0/sqrt(n_k)) for z in zip(query,key,value)]...);
    # once again projected    
    return mha.W(o);
end

function (mha::MultiHeadedAttention)(q,k,v,mask)

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_k   = (size(q,2)//mha.n_heads).num;
    h = 1;
    query  = [view(Q, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    key    = [view(K, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    value  = [view(V, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];

    # each position in the decoder attends to all positions in the decoder up to and including that position
    # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
    # of the softmax which correspond to illegal connections
    w = size(q,1);
    type = eltype(q.data);
    mask1 = tril( fill(type(1), w,w))
    mask2 = triu( fill(type(-1e9),w,w),1)

    o = hcat( [attention(z[1],z[2], z[3],1.0/sqrt(n_k),mask1,mask2) for z in zip(query,key,value)]...);
    # once again projected    
    return mha.W(o);
end


@Flux.treelike(MultiHeadedAttention)