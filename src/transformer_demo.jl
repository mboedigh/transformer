# 
using Statistics
using LinearAlgebra
import Flux
using Transformers

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


# Output
function greedy_decode( model, datum, start_symbol)
    curmode = setmode(model,false)
    memory = encode(model, datum);
    d_vocab = size(model.target_embedding.W,1);
    ys = similar(datum);
    ys[1] = start_symbol;
    for i in 2:length(datum)
        out = decode( model, ys[i-1], memory ); # predict next word based decoding of current word, and memory from encoding
        yhat = model.generator( out );
        word = Flux.onecold(yhat);
        ys[i] =  word[1] # set next word. TODO: check this is right word -- I think it is always a single element, but be sure
    end
    setmode(model,curmode)
    return ys
end

# optimizer
learn_rate(step, warmup=4000, d_model=512) = (d_model.^-0.5f0) .* min.( step.^-0.5f0 , step .* warmup.^-1.5)
opt = Flux.ADAM( learn_rate(1), (0.9, 0.98) )

loss(ypred, y, d_vocab) = Flux.crossentropy( ypred, Flux.onehotbatch(y, 1:d_vocab)' );

function make_transformer_model()
    d_strlen = 10;   # maximum sequence length
    d_vocab = 11;
    d_model = 512;
    n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
    n_layers = 6;    # 6 in the paper
    P_DROP = 0.1;    

    model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = n_layers, n_heads = n_heads);
end

function transformer_loss( model, datum, target, target_y)
    yhat   = model(datum, target);
    d_vocab = size(model.source_embedding.W,1);
    return  Flux.crossentropy( yhat, Flux.onehotbatch(target_y, 1:d_vocab) );
end

function transformer_batch(model, batch, target, target_y)
    memory   = [encode(model,c) for c in Rows(batch)];
    out      = [decode( model, x, memory) for (x, memory) = zip(Rows(target), memory ) ];
    yhat     = [model.generator( o ) for o in out];
    d_vocab  = size(model.source_embedding.W,1);
    l = [ loss( ypred, y, d_vocab) for (ypred, y) in zip(yhat, Rows(target_y))];
    lbar =  mean( l );
end

function transformer_demo()
    d_strlen = 10;   # maximum sequence length
    d_vocab = 11;
    d_model = 512;
    n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
    n_layers = 2;
    P_DROP = 0.1;    
    n_batches = 20;
    batch_size = 30;
    warmup = 400;  # for learning rate in optimiser

    model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = 2);
    setmode(model,true); 

    ps = Flux.params(model);
    local steps = 1;
    my_loss(batch, target, target_y) = transformer_batch(model, batch, target, target_y);
    opt = Flux.ADAM( learn_rate(steps, warmup), (0.9, 0.98) );
    for epoch in 1:10
        total_tokens = 0
        total_loss = 0
        dataset = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);
        tokens_batch = (batch_size)*(d_strlen-1)

        for (batch_num, batch) in enumerate(dataset)
            batch_start = time();
            print( "Epoch $epoch: ");
            target = view(batch, :, 1:size(batch,2)-1);
            target_y = view(batch, :, 2:size(batch,2));
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

    end
    return model
end

