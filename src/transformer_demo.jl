# 
using Statistics
using LinearAlgebra
import Flux
using Transformers

"""
Returns a trained Transformer model on a trivial task that can predict the next token of input sequence (producing an exact copy) 
Based on [Annotated Transfomer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and the 
paper [Attention is all you need](http://arxiv.org/abs/1706.03762)
    
    julia> include("transformer_demo.jl")
    julia> model = transformer_demo();
    julia> @assert all( model.predict(1:10) .= 1:10 ); # if it converged

"""
function transformer_demo()
    d_strlen = 10;   # maximum sequence length
    d_vocab = 11;
    d_model = 512;
    n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
    n_layers = 2;
    P_DROP = 0.1;    
    n_batches = 20;
    batch_size = 30;
    step = 1

    model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = 2);
    ps = Flux.params(model);
        
    # optimizer
    warmup = 400;  # ramp up learning rate over 400 steps. Then decay as shown in learn_rate below 

    opt = Flux.ADAM( learn_rate(step, warmup), (0.9, 0.98) );
    for epoch in 1:30
        dataset = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);
        # global step = transformer_epoch(model, dataset, opt, ps, epoch, step); # this to manually extend epochs from the command line (send the for loop to the REPL)
        step = transformer_epoch(model, dataset, opt, ps, epoch, step); # this works in the script, but not on the command line
    end

    # a = predict(model, 1:10 );
    # @assert all( a .== 1:10);  # this is not deterministic, and so is commented out. Do it with the model that transformer_demo() returns
    return model
end

learn_rate(step, warmup=4000, d_model=512) = (d_model.^-0.5f0) .* min.( step.^-0.5f0 , step .* warmup.^-1.5);

function data_gen(batch_size, d_vocab, d_strlen, d_model, nbatches)
    # "Generate random data for a src-tgt copy task. i.e. the target is an exact copy of the source"
    data = [ rand(1:d_vocab, batch_size, d_strlen ) for in in 1:nbatches]
    # tgt = copy(data);
    # src_mask = [ones( 1, d_model) for i in 1:nbatches]
    # tgt_mask = [tril( ones(d_model,d_model)) for i in 1:nbatches]

    for i = 1:size(data,1)
       d = view( data[i], :, 1 ); 
       fill!(d, 1);
    end
    return data
end

function make_transformer_model()
    d_strlen = 10;   # maximum sequence length
    d_vocab = 11;
    d_model = 512;
    n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
    n_layers = 6;    # 6 in the paper
    P_DROP = 0.1;    

    model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = n_layers, n_heads = n_heads);
end

# process one epoch of data (sets of batches)
function transformer_epoch(model, dataset, opt, ps, epoch, steps)
    total_tokens = 0
    total_loss = 0
    batch_size = size(dataset[1],1)
    d_strlen = size(dataset[1],2)
    tokens_batch = (batch_size)*(d_strlen-1)
    my_loss(batch, target, target_y) = transformer_batch(model, batch, target, target_y);
    
    for (batch_num, batch) in enumerate(dataset)
        batch_start = time();
        print( "Epoch $epoch: ");
        target = view(batch, :, 1:size(batch,2)-1);  # the target is a copy of the input without the last token
        target_y = view(batch, :, 2:size(batch,2)); # the target_y is what we are predicting (the next token in the input)
        opt.eta = learn_rate(steps,400);
        data = [(batch, target, target_y)];
        Flux.train!(my_loss, ps, data, opt)
        
        lbar = my_loss( batch, target, target_y);
        total_tokens += tokens_batch;
        rate = tokens_batch/(time() - batch_start);
        
        s = Base.Printf.@sprintf( "Batch: %d Step %d: learn_rate: %.6f, batch_loss: %.2f, tokens: %d, token/s: %.2f",
        batch_num, steps, opt.eta, lbar, total_tokens, rate )
        println( s );
        steps += 1; 
    end
    
    return steps;
end


loss(ypred, y, d_vocab) = Flux.crossentropy( ypred, Flux.onehotbatch(y, 1:d_vocab)' );

# calculate mean loss over a batch of data
function transformer_batch(model, batch, target, target_y)
    memory   = [encode(model,c) for c in Rows(batch)];
    out      = [decode( model, x, memory) for (x, memory) = zip(Rows(target), memory ) ];
    yhat     = [model.generator( o ) for o in out];
    d_vocab  = size(model.source_embedding.W,1);
    l = [ loss( ypred, y, d_vocab) for (ypred, y) in zip(yhat, Rows(target_y))];
    lbar =  mean( l );
end

# the paper uses smoothed loss function, which I did not implement
function transformer_loss( model, datum, target, target_y)
    yhat   = model(datum, target);
    d_vocab = size(model.source_embedding.W,1);
    return  Flux.crossentropy( yhat, Flux.onehotbatch(target_y, 1:d_vocab) );
end

struct Rows
    A::AbstractArray
end
Base.iterate(r::Rows) =  size(r.A,1)>0 ? (r.A[1,:],2) : nothing
Base.iterate(r::Rows, state) =  state <= size(r.A,1) ? (r.A[state,:], state+1) : nothing
Base.length(r::Rows) = size(r.A,1)
Base.eltype(r::Rows) = eltype(r::A)


struct Cols
    A::AbstractArray
end
Base.iterate(r::Cols) =  size(r.A,2)>0 ? (r.A[:,1],2) : nothing
Base.iterate(r::Cols, state) =  state <= size(r.A,2) ? (r.A[:,state], state+1) : nothing
Base.length(r::Cols) = size(r.A,2)
Base.eltype(r::Cols) = eltype(r::A)