#
using Statistics
using LinearAlgebra
import Flux
using  Transformers
using Flux: gradient

"""
Returns a trained Transformer model on a trivial task that can predict the next token of input sequence (producing an exact copy)
Based on [Annotated Transfomer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and the
paper [Attention is all you need](http://arxiv.org/abs/1706.03762)

    julia> cd("path to folder with transformer_demo.jl")
    julia> push!(LOAD_PATH, ".")
    julia> include("transformer_demo.jl")
    julia> model = transformer_demo();
    julia> @assert all( model.predict(1:10) .= 1:10 ); # true if it converged

"""
function transformer_demo( ;max_seqlen=10, d_vocab=11, d_model=512, n_heads=8,
                            n_layers=2, p_drop = 0.01f0, n_batches = 20, batch_size = 30, n_epochs = 10)

    model = Transformer(max_seqlen=max_seqlen, d_vocab=d_vocab, d_model=d_model, p_drop = p_drop, n_layers = n_layers);
    dataset = data_gen( batch_size, d_vocab, max_seqlen, n_batches);

    run_transformer( model, dataset, n_epochs);

   # a = predict(model, 1:10 );
   # @assert all( a .== 1:10);  # this is not deterministic, and so is commented out. Do it with the model that transformer_demo() returns
end

# generate data for the copy task
function data_gen(batch_size=20, d_vocab=11, max_seqlen=10, n_batches = 20)
    # "Generate random data for a src-tgt copy task. i.e. the target is an exact copy of the source"
    data = [ rand(1:d_vocab, batch_size, max_seqlen ) for in in 1:n_batches]
    # tgt = copy(data);
    # src_mask = [ones( 1, d_model) for i in 1:nbatches]
    # tgt_mask = [tril( ones(d_model,d_model)) for i in 1:nbatches]

    # start of sequence token (I'm not sure why this is needed, but it is in "Annotated Transfomer" )
    for i = 1:size(data,1)
        d = view( data[i], :, 1 );
        fill!(d, 1);
    end

    d = map( d->(d,view(d,:,1:size(d,2)-1), view(d,:, 2:size(d,2))), data)
    return d
end

# train, or continue training, the transformer model over the dataset n_epoch (more) times.
# currentlyk resets the learning rate
function run_transformer( model, dataset, n_epochs = 10; first_step=1)
    # optimizer
    warmup = 400;  # ramp up learning rate over 400 steps. Then decay as shown in learn_rate below
    stepnum = first_step;
    opt = Flux.ADAM( learn_rate(stepnum, warmup), (0.9, 0.98) )
    ps = Flux.Params(Flux.params(model));
    for epoch in 1:n_epochs
        stepnum = transformer_epoch(model, dataset, opt, ps, epoch, stepnum); # this works in the script, but not on the command line
    end

    return model # i don't think this is necessary, becuase the model was passed by reference
end

function transformer_hparams_tiny()
    Dict(
    :max_seqlen => 10,   # maximum sequence length
    :d_vocab => 11,
    :d_model => 64,
    :n_heads => 4,    # number of heads in Mulit-headed attention (8 were used in the paper)
    :n_layers => 2,    # 6 layers were used in both the encoder and decoder stacks
    :p_drop => 0.10f0,
    )
end

function transformer_demo_tiny()
    transformer_demo( ; n_batches=20, batch_size=30, transformer_hparams_tiny()... );
end

learn_rate(stepnum, warmup=4000, d_model=512) = (d_model.^-0.5f0) .* min.( stepnum.^-0.5f0 , stepnum .* warmup.^-1.5);


function data_gen_repeated_numbers(;batch_size=30, d_vocab=11, input_seqlen = 10, max_seqlen=15, n_batches=20)
    eos = d_vocab ;
    function gen_input(d_vocab, input_seqlen, batch_size)
        input = rand( 1:d_vocab, batch_size, input_seqlen);
    end
    function gen_batch(batch_size, d_vocab, input_seqlen)
        input = gen_input(d_vocab, input_seqlen, batch_size)
        target   = Array{Array{eltype(input),1},1}(undef, size(input,1));
        target_y = Array{Array{eltype(input),1},1}(undef, size(input,1));
        for (i, x) = enumerate(eachrow(input))
            all_twos    = findall(x.==2);
            all_threes  = findall(x.==3);
            k = sort([vec(LinearIndices(x)); all_twos; all_threes ])'
            targets = input[k];
            target[i] = targets[1:end-1];
            target_y[i] = targets[2:end];
        end
        return (input, target, target_y);
    end
    dataset = [ gen_batch( batch_size, d_vocab, input_seqlen) for i in 1:n_batches];
end

# process one epoch of data (sets of batches)
function transformer_epoch(model, dataset, opt, ps, epoch, stepnum)
    total_tokens = 0
    total_loss = 0
    batch_size = size(dataset[1][1],1)
    eos = size(model.source_embedding.W,1)
    my_loss(batch, target, target_y) = transformer_batch(model, batch, target, target_y);
    for (batch_num, batch) in enumerate(dataset)
        batch_start = time();
        print( "Epoch $epoch: ");
        opt.eta = learn_rate(stepnum,400);
        lbar = my_loss( batch...);
        Flux.back!(lbar)
        gs = Flux.Tracker.Grads()
        for p in ps
            gs[Flux.Tracker.tracker(p)] = Flux.Tracker.extract_grad!(p)
        end
        Flux.Tracker.update!(opt, ps, gs);
        
        tokens_batch = sum(batch[3] .!= eos)
        total_tokens += tokens_batch;
        rate = tokens_batch/(time() - batch_start);

        s = Base.Printf.@sprintf( "Batch: %d Step %d: learn_rate: %.6f, batch_loss: %.2f, tokens: %d, token/s: %.2f",
        batch_num, stepnum, opt.eta, lbar, total_tokens, rate )
        println( s );
        stepnum += 1;
    end
    return stepnum;
end


# average cross entropy per token (each columns of ypred is a token)
loss(ypred, y, d_vocab) = Flux.crossentropy( ypred, Flux.onehotbatch(y, 1:d_vocab)' );

# calculate mean loss over a batch of data
function transformer_batch(model, batch::AbstractMatrix, target::AbstractMatrix, target_y::AbstractMatrix)
    memory   = [encode(model,c) for c in eachrow(batch)];
    out      = [decode( model, x, memory) for (x, memory) = zip(eachrow(target), memory ) ];
    yhat     = [model.generator( o ) for o in out];
    d_vocab  = size(model.source_embedding.W,1);
    l = [ loss( ypred, y, d_vocab) for (ypred, y) in zip(yhat, eachrow(target_y))];
    lbar =  mean( l );
end

# calculate mean loss over a batch (ragged, or variable length target)
function transformer_batch(model, batch, target::AbstractVector, target_y::AbstractVector)
    memory   = [encode(model,c) for c in eachrow(batch)];
    out      = [decode( model, x, memory) for (x, memory) = zip(target, memory ) ];
    yhat     = [model.generator( o ) for o in out];
    d_vocab  = size(model.source_embedding.W,1);
    l = [ loss( ypred, y, d_vocab) for (ypred, y) in zip(yhat, target_y)]; # loss per token for each input sequence
    lbar =  mean( l ); # avg loss
end

# the paper uses smoothed loss function, which I did not implement
function transformer_loss( model, datum, target, target_y)
    yhat   = model(datum, target);
    d_vocab = size(model.source_embedding.W,1);
    return  Flux.crossentropy( yhat, Flux.onehotbatch(target_y, 1:d_vocab) );
end


