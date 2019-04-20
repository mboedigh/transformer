
struct  Embedding
    W;  # nvocab x ndim. initialize with TrackedArray to learn parameters or a regular Array to use without tracking (or training)
end
function Embedding(n_vocab, d_model; init=Flux.glorot_uniform)
    return Embedding(Flux.Tracker.param(init(d_model, n_vocab)))
end
(e::Embedding)(x) = e.W[:,x]
(e::Embedding)(x::Integer)  = e.W[:,x];

@Flux.treelike Embedding