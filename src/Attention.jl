import Flux

# returns the normalized attention score (query_len x key_len) * value matrix
# q,k,v are the attention matrices
# qmask is a query sequence mask. it has 1s in positions to keep and 0s in positions to mask (i.e. padded)
# qmask is a query-length vector that is broadcast over the scores (as if it were repeated for each column of the score matrix )
function attention( q, k, v, qmask, scale::Real, mask=nothing)
    score = scale.*( q*k');
    qmask!=nothing && (score = score .+ qmask);
    mask !=nothing && (score = score .+ mask);
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

function MultiHeadedAttention( n_heads::Integer, d_in::Integer, d_k::Integer; init = Flux.glorot_uniform, initb=Flux.zeros) 
    return MultiHeadedAttention( n_heads, [Linear(d_in, d_k*n_heads, initW=init, initb=initb) for i in 1:4]...);
end

# On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
# These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
# The online version of this (Annotated Transformer uses a gain and bias)

# query, key, value, mask (additive to presoftmax q*v' matrix with zeros and -Inf)
# hide is only true under self_attention in decoder. That is always a square mask for a square attention score matrix
function (mha::MultiHeadedAttention)(q,k,v,mask=nothing,hide=false)
    # println("single sequence MHA 3")

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_k   = (size(q,2)//mha.n_heads).num;

    query  = [view(Q, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    key    = [view(K, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];
    value  = [view(V, :, (h*n_k+1):(h+1)*n_k) for h in 0:mha.n_heads-1];

    # each position in the decoder attends to all positions in the decoder up to and including that position
    # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
    # of the softmax which correspond to illegal connections
    T = Float32;
    scale = T(1.0/sqrt(n_k))
    fmask = nothing; # mask future tokens 
    if hide
        w = size(q,1);
        fmask = triu( fill(T(-1e9),w,w),1);
    end

    qmask = nothing
    if (mask != nothing)
        mask = vec(mask');
        qmask = zeros(T,size(mask));
        qmask[mask .== 0] .= T( -1e9 ); #TODO: -Inf is guaranteed to work with any type but -1e9 is not, but Softmax fails
    end

    o = map(z->attention(z..., qmask, scale, fmask), zip(query,key,value));
    o = hcat(o...); 
    return mha.W(o);
end
(mha::MultiHeadedAttention)(q) = mha(q,q,q); # one argument call for self-attention without masking

# Batch version
# query, key, value, mask for query sequence (vector added (broadcast) to presoftmax q*v' matrix with zeros and -Inf)
# hide_future_tokens prevents a token at position i in the target from seeing tokens > i. It is used during self_attention
# q,k,v and output are arrays of size seqlen*n_sequences x d_model
function (mha::MultiHeadedAttention)(q,k,v,num_seqs::Int, mask=nothing,hide_future_tokens=false)
#    println("batched sequence MHA")
    n_s = num_seqs;             # number of sequences
    q_len = Int(size(q,1)/n_s);   # target (query) sequence length (tokens). All, possibly padded, sequences must be the same length
    @assert size(q,1) % q_len == 0

    k_len = Int(size(k,1)/n_s);   # souce (key & value) sequence length (tokens). padding not yet supported
    @assert size(k,1) % k_len == 0

    Q,K,V = mha.Q(q), mha.K(k), mha.V(v);
    n_h   = mha.n_heads;           # number of heads
    d_h   = (size(q,2)//n_h).num;  # dimension of each head

    qkv  = nothing;
    T    = Float32; # TODO make this part of the Transformer Spec and apply it everywhere to avoid mixed floating modes
    if mask == nothing
        qkv = [(view(Q, (i*q_len+1):(i+1)*q_len, (h*d_h+1):(h+1)*d_h), 
                view(K, (i*k_len+1):(i+1)*k_len, (h*d_h+1):(h+1)*d_h), 
                view(V, (i*k_len+1):(i+1)*k_len, (h*d_h+1):(h+1)*d_h), 
                nothing)  for i in 0:n_s-1, h in 0:n_h-1];
    else
        mask = vec(mask');
        qmask = zeros(T,size(mask));
        qmask[mask .== 0] .= T( -1e9 ); #TODO: -Inf is guaranteed to work with any type but -1e9 is not, but Softmax fails with -Inf
        qkv = [(view(Q, (i*q_len+1):(i+1)*q_len, (h*d_h+1):(h+1)*d_h), 
                view(K, (i*k_len+1):(i+1)*k_len, (h*d_h+1):(h+1)*d_h), 
                view(V, (i*k_len+1):(i+1)*k_len, (h*d_h+1):(h+1)*d_h), 
                view( qmask, (i*q_len+1):(i+1)*q_len))  for i in 0:n_s-1, h in 0:n_h-1];
    end

    # each position in the decoder attends to all positions in the decoder up to and including that position
    # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
    # of the softmax which correspond to illegal connections
    scale = T(1.0/sqrt(d_h))
    o = nothing;
    future_mask = nothing;
    if hide_future_tokens
        future_mask = triu( fill(T(-1e9),q_len,q_len),1);
    end
    o = map( z->attention(z..., scale, future_mask), qkv); 
    # all the arrays in o are in the right position, but there is no way to flatten o
    # there is also no way to use a view into preallocated memory and set the contents direction with calls to attention! (if it existed)
    #t1 = vcat( o...);   # stack all scores from entire batch from head 1, then head 2 etc
    #t2 = reshape( t1, (d_s, n_s, n_h, :) ); # shaped as: position x sequence x head x feature_within_head
    #t3 = permutedims( t2, (1,2,4,3));       # permute to: position x feature x sequence x head
    #t4 = reshape( t3, (d_s*n_s, d_h*n_h));  # 
    o = reshape( permutedims( reshape( vcat(o...), (q_len, n_s, n_h, :) ), (1,2,4,3)), (q_len*n_s, d_h*n_h));  # 
    return mha.W(o);

end

@Flux.treelike(MultiHeadedAttention)