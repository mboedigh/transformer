
import Flux

function attention( q, k, v, scale, mask=nothing)
    score = scale.*( q*k')
    mask!=nothing && (score = score .+ mask)
    Flux.softmax(score')'*v  
end

# Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, 
# keys and values h times with different, learned linear projections to dq, dk and dv dimensions, respectively. 
# I am doing these all in one matrix multiply
struct MultiHeadedAttention
    n_heads::Integer
    Q::Linear
    K::Linear
    V::Linear
    # On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
    # These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
    W::Linear
end

function MultiHeadedAttention( n_heads::Integer, d_in::Integer, d_k::Integer; init = Flux.glorot_uniform) 
    return MultiHeadedAttention( n_heads, [Linear(d_in, d_k*n_heads, initW=init, initb=init) for i in 1:4]...);
end

# On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
# These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
# The online version of this (Annotated Transformer uses a gain and bias)

# query, key, value, mask (additive to presoftmax q*v' matrix with zeros and -Inf)
# hide is only true under self_attention in decoder. That is always a square mask for a square attention score matrix
function (mha::MultiHeadedAttention)(q,k,v,mask=nothing,hide=false)

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_k   = (size(q,2)//mha.n_heads).num;

    query  = [view(Q, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    key    = [view(K, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    value  = [view(V, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];

    # mask unused positions in query
    mask!=nothing && (mask = repeat( mask, 1, size(k,1)));

    # each position in the decoder attends to all positions in the decoder up to and including that position
    # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
    # of the softmax which correspond to illegal connections
    scale = typeof(q.data[1])(1.0/sqrt(n_k))
    o = nothing;
    if hide
        w = size(q,1);
        type = eltype(q.data);
        if (mask==nothing)
            mask = triu( fill(type(-1e9),w,w),1);
        else
            mask .+= triu( fill(type(-1e9),w,w),1);
        end
    end
    o = [attention(z[1],z[2], z[3],scale,mask) for z in zip(query,key,value)];
    o = hcat(o...); # supposedly slower than reduct(hcat,o), but produces different outputs
    # o = reduce( hcat, o); # avoids splat operator (o = hcat(o...)), which is supposedly slower
    # once again projected    
    return mha.W(o);
end
(mha::MultiHeadedAttention)(q) = mha(q,q,q); # one argument call for self-attention without masking


# Batch version
# query, key, value, mask (additive to presoftmax q*v' matrix with zeros and -Inf)
# hide is only true under self_attention in decoder. That is always a square mask for a square attention score matrix
# q,k,v and output are arrays of size seqlen*n_sequences x d_model
# input m
function (mha::MultiHeadedAttention)(q,k,v,num_seqs::Int, mask=nothing,hide=false)
    n_s = num_seqs;             # number of sequences
    d_s = Int(size(q,1)/n_s);   # sequence length (tokens). All, possibly padded, sequences must be the same length
    @assert size(q,1) % d_s == 0

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_h   = mha.n_heads;           # number of heads
    d_h   = (size(q,2)//n_h).num;  # dimension of each head

    QKV  = [(view(Q, (i*d_s+1):(i+1)*d_s, (h*d_h+1):(h+1)*d_h), 
             view(K, (i*d_s+1):(i+1)*d_s, (h*d_h+1):(h+1)*d_h), 
             view(V, (i*d_s+1):(i+1)*d_s, (h*d_h+1):(h+1)*d_h))  for i in 0:n_s-1, h in 0:n_h-1];

    # mask unused positions in query sequence
    mask!=nothing && (mask = repeat( mask, 1, size(k,1)));

    # each position in the decoder attends to all positions in the decoder up to and including that position
    # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
    # of the softmax which correspond to illegal connections
    scale = eltype(q.data)(1.0/sqrt(d_h))
    o = nothing;
    if hide
        w = d_s;
        type = eltype(q.data);
        if (mask==nothing)
            mask = triu( fill(type(-1e9),w,w),1);
        else
            mask .+= triu( fill(type(-1e9),w,w),1);
        end
    end
    o = [attention(z[1],z[2], z[3],scale,mask) for z in QKV];
    # all the arrays in o are in the right position, but there is no way to flatten o
    # there is also no way to use a view into preallocated memory and set the contents direction with calls to attention! (if it existed)
    #t1 = vcat( o...);   # stack all scores from entire batch from head 1, then head 2 etc
    #t2 = reshape( t1, (d_s, n_s, n_h, :) ); # shaped as: position x sequence x head x feature_within_head
    #t3 = permutedims( t2, (1,2,4,3));       # permute to: position x feature x sequence x head
    #t4 = reshape( t3, (d_s*n_s, d_h*n_h));  # 
    o = reshape( permutedims( reshape( vcat( o...), (d_s, n_s, n_h, :) ), (1,2,4,3)), (d_s*n_s, d_h*n_h));  # 
    return mha.W(o);

end

@Flux.treelike(MultiHeadedAttention)