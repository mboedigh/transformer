using Flux

struct  Embedding
    W;  # nvocab x ndim. initialize with TrackedArray to learn parameters or a regular Array to use
end
function Embedding(n_vocab, d_model; init=Flux.glorot_uniform)
    return Embedding(Flux.Tracker.param(init(n_vocab, d_model)))
end

(e::Embedding)(x::Integer)  = reshape(e.W[x,:], 1, : )
(e::Embedding)(x::AbstractVector) = e.W[x,:]
(e::Embedding)(x::AbstractArray) = e.W[vec(x'),:]

@Flux.treelike Embedding