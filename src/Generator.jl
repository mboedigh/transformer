
struct  Generator
    W;  # nvocab x ndim. initialize with TrackedArray to learn parameters or a regular Array to use
end
function Generator( d_model, n_vocab; init=Flux.glorot_uniform)
    return Generator(Flux.Dense( d_model, n_vocab, initW=init))
end

# output is n_vocab x target_seq_len matrix. Columns sum to 1
(g::Generator)(x)  = Flux.softmax( g.W(x') )

@Flux.treelike Generator;