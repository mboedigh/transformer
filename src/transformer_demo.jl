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
    memory = encode(model, datum);
    d_vocab = size(model.target_embedding.W,1);
    ys = similar(datum);
    ys[1] = start_symbol;
    for i in 2:length(datum)
        out = decode( model, ys[i-1], memory ) # predict next word based decoding of current word, and memory from encoding
        yhat = model.generator( out )
        word = Flux.onecold(yhat,d_vocab);
        ys[i] =  word[0] # set next word. TODO: check this is right word -- I think it is always a single element, but be sure
    end
    return ys
end

# optimizer
learn_rate(step, warmup=4000, d_model=512) = (d_model.^-0.5f0) .* min.( step.^-0.5f0 , step .* warmup.^-1.5)
opt = Flux.ADAM( learn_rate(1), (0.9, 0.98) )

loss(ypred, y, d_vocab) = Flux.crossentropy( ypred, Flux.onehotbatch(y, 1:d_vocab) );

function transformer_demo()
    d_strlen = 10;   # maximum sequence length
    d_vocab = 11;
    d_model = 512;
    n_heads  = 8;    # number of heads in Mulit-headed attention (8 were used in the paper)
    n_layers = 6;
    P_DROP = 0.1;    

    model = Transformer(d_strlen, d_vocab, d_model, p_drop = P_DROP, n_layers = 2);
    setmode(model,true); 

    ps = Flux.params(model);
    warmup = 400;
    local steps = 1;
    local failed = false
    local cur_data
    for epoch in 1:10
        total_tokens = 0
        total_loss = 0
        n_batches = 20;
        batch_size = 30;
        batches = data_gen( batch_size, d_vocab, d_strlen, d_model, n_batches);

        batch = batches[1]  # for debugging at REPL
        datum = batch[1,:]
        target = datum[1:end-1]
        target_y = datum[2:end]
        
        l = 0;
        start = time();
        for (batch_num, batch) in enumerate(batches)
            print( "Epoch $epoch: ");
            b = batch;
            cur_data = b;
            t = b[:, 1:end-1];
            t_y = b[:, 2:end];
            m   = [encode(model,c) for c in Rows(b)];
            out = [decode( model, x, memory) for (x, memory) = zip(Rows(t), m ) ];
            yhat = [model.generator( o ) for o in out];
            
            l = [ loss( ypred, y, d_vocab) for (ypred, y) in zip(yhat, Rows(t_y))];
            lbar =  mean( l );
            if (isnan(lbar) || isinf(lbar))
                failed = true;
                println("Failed");
                break;
            end
            Flux.back!(lbar);
            opt = Flux.ADAM( learn_rate(steps, 400), (0.9, 0.98) );
            Flux.Optimise.update!(opt, ps);
            steps += batch_size;

            total_tokens += length(t_y);
            elapsed = time() - start;
            rate = total_tokens/elapsed;
            println( "Batch: $batch_num Step $steps: learn_rate = $(opt.eta), batch_loss = $lbar, token/s = $rate");
        end

        failed && break;
    end
    return (model, cur_data);
end

