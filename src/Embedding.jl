using Flux

struct  Embedding
    W;  # nvocab x ndim. initialize with TrackedArray to learn parameters or a regular Array to use
end
function Embedding(n_vocab, d_model; init=Flux.glorot_uniform)
    return Embedding(Flux.Tracker.param(init(n_vocab, d_model)))
end

# gather(w::TrackedArray, xs::AbstractVector{Int}) = Flux.Tracker.track(gather, w, xs)  # intercept "real" gather function as per Flux help on custom gradients
# gather(w::AbstractMatrix{T}, xs::AbstractVector{Int}) where T = w[xs,:]   # "real" gather returns parameter vector by index
# function ∇gather(Δ::AbstractMatrix{T}, w::AbstractMatrix{T}, xs::AbstractVector{Int}) where T  # back
#     ys = fill!(similar(w), zero(T))
#     ys[xs,:] .+= Δ

#     return ys
# end

# @Flux.Tracker.grad gather(w::TrackedArray, xs) = gather(Flux.Tracker.data(w), xs), Δ->(∇gather(Δ, Flux.Tracker.data(w), xs),nothing)

# (e::Embedding)(x) = gather( e.W, x);
(e::Embedding)(x) = e.W[x,:]
(e::Embedding)(x::Integer)  = reshape(e.W[x,:], 1, : )

@Flux.treelike Embedding