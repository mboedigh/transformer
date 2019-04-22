
import Flux

# input d_attn x d_seqlen tensors, q,k,v
# output size(input)
function attention( q, k, v, scale)
    score = scale .* (k' * q)  # scale is reportedly for numerical stability
    v * Flux.softmax(score)
end

# input d_attn x d_seqlen tensors, q,k,v. mask1 and mask2 are d_seqlen x d_seqlen upper and lower tri masks. scale is scalar
# output size(input)
function attention( q, k, v, scale, mask1, mask2)
    score = scale .* (k' * q).*mask1 .+ mask2  # scale is reportedly for numerical stability
    v * Flux.softmax(score)
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
    return MultiHeadedAttention( n_heads, [Flux.Dense(d_in, d_k*n_heads, initW=init, initb=init) for i in 1:4]...);
end

# On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values.
# These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
# The online version of this (Annotated Transformer uses a gain = bias)
function (mha::MultiHeadedAttention)(q,k,v,mask=nothing)

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_k   = (size(q,1)//mha.n_heads).num;

    query  = [view(Q, (h*n_k+1):(h+1)*n_k,:) for h in 0:mha.n_heads-1];
    key    = [view(K, (h*n_k+1):(h+1)*n_k,:) for h in 0:mha.n_heads-1];
    value  = [view(V, (h*n_k+1):(h+1)*n_k,:) for h in 0:mha.n_heads-1];

    # each position in the decoder attends to all positions in the decoder up to and including that position
    # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
    # of the softmax which correspond to illegal connections
    scale = typeof(q.data[1])(1.0/sqrt(n_k))

    if (mask != nothing)
        w     = size(q,2);    # seq_len
        type  = eltype(q.data);
        mask1 = triu( fill(type(1), w,w))
        mask2 = tril( fill(type(-Inf),w,w),-1)  # try -Inf instead of -1e9 (-inf is generalizes to other Float types)
        o = [attention(z[1],z[2], z[3],scale,mask1,mask2) for z in zip(query,key,value)];
    else
        o = [attention(z[1],z[2], z[3],scale) for z in zip(query,key,value)];
    end

    # These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
    o = vcat(o...); # supposedly slower than reduce(vcat,o), but produces different outputs
    # o = reduce( hcat, o); # avoids splat operator (o = vcat(o...)), which is supposedly slower
    # once again projected
    return mha.W(o);
end

(mha::MultiHeadedAttention)(q) = mha(q,q,q); # one argument call for self-attention without masking

@Flux.treelike(MultiHeadedAttention)
