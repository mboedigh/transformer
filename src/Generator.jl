
struct  Generator
    W::Linear;  # nvocab x ndim. initialize with TrackedArray to learn parameters or a regular Array to use
end
function Generator( d_model, d_vocab; init=Flux.glorot_uniform)
    return Generator(Linear( d_model, d_vocab, initW=init))
end

# output is n_vocab x target_seq_len matrix. Columns sum to 1
(g::Generator)(x)  = Flux.logsoftmax( g.W(x)' )'
# (g::Generator)(x)  = g.W(x)

@Flux.treelike Generator;