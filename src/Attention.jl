
import Flux

struct Attention
    Q
    K
    V
    scale
end

# Attention
# d_in is the input dimension - e.g. d_model (same as word embedding size at first layer), 
# d_k is the output dimension for each attention node (e.g. d_model/n_head)
# "To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512."
function Attention( d_in::Integer, d_k::Integer; init = Flux.glorot_uniform)
    return Attention( Flux.param(init(d_in, d_k)), 
                      Flux.param(init(d_in, d_k)),
                      Flux.param(init(d_in, d_k)), 
                      1.0f0 ./ sqrt(d_k) )
end
@Flux.treelike Attention

function (z::Attention)(q, k, v, mask::Bool = false) 
    # score each position in the sequence versus all others
    # rows correspond to query positions and columns to keys
    score = z.scale.*( (q*z.Q)*(k*z.K)') 
    if mask
        # each position in the decoder attends to all positions in the decoder up to and including that position
        # we need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
        # We implement this inside the scaled dot-product attention by masking out (setting to -inf) all values in the input
        # of the softmax which correspond to illegal connections
        w = size(score,1);
        M = similar(score)
        for i in 1:w
            for j in 1:w
                if (j > i)
                    M[i,j] = score[i,j] < 0 ? eltype(q.data)(Inf) : eltype(q.data)(-Inf)
                else
                    M[i,j] = one(eltype(q.data));
                end
            end;
        end
        score = score.*mask
       end
    return Flux.softmax(score')'*(v*z.V)  
end

# Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, 
# keys and values h times with different, learned linear projections to dq, dk and dv dimensions, respectively. 
struct MultiHeadedAttention
    heads::Array{Attention,1}
    # On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
    # These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
    W;
end
function MultiHeadedAttention( n_heads::Integer, d_in::Integer, d_k::Integer; init = Flux.glorot_uniform) 
    return MultiHeadedAttention( [Attention(d_in, d_k, init=init) for i in 1:n_heads], Flux.param(init(n_heads*d_k,n_heads*d_k )) );
end

# On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
# These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
(mha::MultiHeadedAttention)(q,k,v,mask=false) = hcat([mha.heads[i](q,k,v, mask) for i in 1:length(mha.heads)]...)*mha.W;

Flux.children(c::MultiHeadedAttention) = (c.heads..., c.W)
Flux.mapchildren(f, c::MultiHeadedAttention) = MultiHeadedAttention( f.(c.heads..., c.W)...)