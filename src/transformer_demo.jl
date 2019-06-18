#
using Statistics
using LinearAlgebra
using Flux, Zygote
using Transformers
# using Flux: gradient

init_pe( n, d_model ) = Transformers.PositionalEncoding( 2048, d_model, 0.0f0 )( zeros(n, d_model));

"""
    transformer_demo()
Returns a trained Transformer model on a trivial task that can predict the next token of input sequence (producing an exact copy)
Based on [Annotated Transfomer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and the
paper [Attention is all you need](http://arxiv.org/abs/1706.03762)

    julia> cd("path to folder with transformer_demo.jl")
    julia> push!(LOAD_PATH, ".")
    julia> include("transformer_demo.jl")
    julia> model = transformer_demo();
    julia> @assert all( model.predict(1:10) .= 1:10 ); # true if it converged

    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_stutter_task( ;n_batches=90, batch_size=30 );
    julia> train_transformer!( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)

"""
function transformer_demo( ;max_seqlen=1024, d_vocab=13, d_model=512, n_heads=8,
                            n_layers=2, p_drop = 0.01f0, n_batches = 20, batch_size = 30, n_epochs = 10)

    model   = Transformer(; max_seqlen=max_seqlen, d_vocab=d_vocab, d_model=d_model, p_drop = p_drop, n_layers = n_layers);
    model = Flux.mapleaves(Flux.data, model); # for use with Zygote, which wants raw data, not Tracked arrays
    dataset = data_gen_copy_task( batch_size=batch_size, d_vocab=d_vocab, seqlen=12, n_batches=n_batches);
    train_transformer!( model, dataset, n_epochs);

   # a = predict(model, 1:10 );
   # @assert all( a .== 1:10);  # this is not deterministic, and so is commented out. Do it with the model that transformer_demo() returns
end

"""
    data_gen_stutter_task(;batch_size=20, d_vocab=13, seqlen=12, n_batches = 30) 
Returns a dataset for stutter task. dataset is a collection of batches. seqlen between batches will vary from 3 to seqlen (before special tokens are added)
this task tests batches of variable sequence length (each batch has uniform length sequences)
this task tests target sequence padding, since targets are of variable length
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_stutter_task( ;n_batches=90, batch_size=30 );
    julia> train_transformer!( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_stutter_task(;batch_size=20, d_vocab=13, seqlen=12, n_batches = 30) 
    seqlens = rand(4:seqlen, n_batches);
    [data_gen_batch( ()->data_gen_stutter_pair(d_vocab, seqlens[i]), batch_size ) for i in 1:n_batches]
end

"""
simplest task. target is an exact copy of the source
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_copy_task( ;n_batches=90, batch_size=30 );
    julia> train_transformer!( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_copy_task(;batch_size=20, d_vocab=13, seqlen=12, n_batches = 30)    
    seqlens = rand(4:seqlen, n_batches);
    [data_gen_batch( ()->data_gen_copy_pair(d_vocab, seqlens[i]), batch_size)  for i in 1:n_batches]
end

"""
encode variable length sources (batches contain the sequences of the same length)
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_dyslexic_task( ;n_batches=90, batch_size=30 );
    julia> train_transformer!( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_dyslexic_task(;batch_size=20, d_vocab=13, seqlen=12, n_batches = 30) 
    seqlens = rand(4:seqlen,n_batches);
    [data_gen_batch( ()->data_gen_dyslexic_pair(d_vocab, seqlens[i]), batch_size)  for i in 1:n_batches]
end

"""
this tests varied meaning of tokens depending on the presence of another token in the same sequence
encode variable length sources (batches contain the sequences of the same length)
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_contextual_task( ;n_batches=90, batch_size=30 );
    julia> train_transformer!( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_contextual_task(;batch_size=20, d_vocab=13, seqlen=12, n_batches = 30) 
    seqlens = rand(4:seqlen,n_batches);
    [data_gen_batch( ()->data_gen_contextual_pair(d_vocab, seqlens[i]), batch_size)  for i in 1:n_batches]
end

# train, or continue training, the transformer model over the dataset n_epoch (more) times.
# currentlyk resets the learning rate
function train_transformer!( model, dataset, n_epochs = 10; stepnum=1)
    # optimizer
    warmup = 400;  # ramp up learning rate over 400 steps. Then decay as shown in learn_rate below
    # opt = Flux.ADAM( learn_rate(stepnum, warmup), (0.9, 0.98) )
    opt = Flux.ADAM();                     # try default parameters
    ps  = Flux.params(model);
    min_loss = Float32(Inf);
    for epoch in 1:n_epochs
        min_loss = transformer_epoch(model, dataset, opt, ps, epoch, stepnum,min_loss=min_loss); # this works in the script, but not on the command line
    end

    return model # i don't think this is necessary, becuase the model was passed by reference
end

function transformer_hparams_tiny()
    Dict(
        :max_seqlen => 1024, # positional encoding size (must be larger than input sequence length)
        :d_vocab => 13,      # total vocab including special "words" for start, stop and unknown
        :d_model => 64,
        :n_heads => 4,     # number of heads in Mulit-headed attention (8 were used in the paper)
        :n_layers => 2,    # In the paper 6 layers were used in both the encoder and decoder stacks
        :p_drop => 0.1f0,
        )
    end
    
# much smaller model but it will still eventually converged on the copy task
function transformer_demo_tiny()
    transformer_demo( ; n_epochs=30, n_batches=20, batch_size=30, transformer_hparams_tiny()... );
end

learn_rate(stepnum, warmup=4000, d_model=512) = (d_model.^-0.5f0) .* min.( stepnum.^-0.5f0 , stepnum .* warmup.^-1.5);

# process one epoch of data (sets of batches)
function transformer_epoch(model, dataset, opt, ps, epoch, stepnum; min_loss = Float32(Inf) )
    total_tokens = 0
    for (batch_num, batch) in enumerate(dataset) 
        batch_start = time();
        print( "Epoch $epoch");
        # opt.eta = learn_rate(stepnum,400);

        # lbar  = transformer_loss(model, batch...);  # call loss function to save the result
        # grads = Zygote.gradient( ()->lbar,model);  
        grads = Zygote.gradient(model) do model
            return lbar  = transformer_loss(model, batch...);  # call loss function to save the result
        end
        # Peel outer Tuple to access gradient of first parameter
        grads = grads[1]

        # Apply recursive update to our model:
        zyg_update!(opt, model, grads)

        # Flux.Optimise.update!(opt, ps, gs);
        
        tokens = sum( batch[2] .!= 3) # sum non-padding target tokens
        total_tokens += tokens;
        rate = tokens/(time() - batch_start);

        lbar  = transformer_loss(model, batch...);  # call loss function to save the result
        lbar < min_loss && (min_loss = lbar);
        s = Base.Printf.@sprintf( ", Batch %d,  learn_rate %.5f,  tokens %d token/s %.1f,  batch_loss %.2f,  min_batch_loss %.2f",
                                 batch_num, opt.eta, tokens, rate, lbar, min_loss );
        println( s );
        x,t = batch[1][1,:], batch[2][1,:];
        labels = t[2:end]; # true answer
        yhat = model( x, t);  # log softmax for predicted tokens
        pred = Flux.onecold(yhat')'[1:end-1]; # predicted token
        n = getmask(t);
        if n != nothing
            pred .*= n[2:end];
        end
        labels =  join(map( x->Base.Printf.@sprintf( "%2d ", x), labels ));
        pred   =  join(map( x->Base.Printf.@sprintf( "%2d ", x), pred ));
        println( labels); # unmasked golden
        println( pred );
        println( transformer_loss( model, x, t));

    end
    return min_loss;
end


"""
generate example datum for the copy task. 
Returns a tuple containing one source and identical target sequence pair. Each sequence contains 
a special start and stop token
the first and last tokens are 1 and 2 and the middle seqlen is randomly generated from 4:d_vocab
"""
function data_gen_copy_pair( d_vocab, seqlen ) 
   d = [1 rand(4:d_vocab, 1, seqlen) 2]
   (d,d)
end

# generate pair of source-target data. The target is the same as the source, except some tokens (5 and 7) are duplicated 
function data_gen_stutter_pair( d_vocab, seqlen)
    x = [1 rand(4:d_vocab, 1, seqlen) 2]
    i = LinearIndices(x);
    i_fives   = findall(x.==5);
    i_sevens  = findall(x.==7);

    k = sort( vcat(vec(i), i[i_fives], i[i_sevens]) )'
    t = x[k];
    return (x,t);
end

# generate pair of source-target data. The target is the same as the source except certain characters are swapped
# if a swappable token (5 or 7) is the last token in the sequence, nothing happens
# if there are multiple swappable tokens in a row, the token after the set of swappable tokens is moved all the way to the start
# otherwise it is a standard swap
function data_gen_dyslexic_pair( d_vocab, seqlen)
    x = [1 rand(4:d_vocab, 1, seqlen) 2]

    t = copy(x);
    for i in seqlen:-1:2
        c = x[i];
        if (c == 5 || c == 7)
            t[i]   = t[i+1];
            t[i+1] = c;
        end
    end
    
    return (x,t);
end

# generate dataset for the copy task 
# a dataset is a collection of batches. Each batch is a tuple of source and target sequences. 
# for the simple copy task these are just two identical matrices. In a real translation task these would 
# need to allow for different lengths
function data_gen_batch( gen_source_target_pair=()->data_gen_copy_pair(13, 12), batch_size=20)
    # "Generate random data for a src-tgt copy task. i.e. the target is an exact copy of the source"

    # add 3 to vocab for start end and unknown symbols and subtract 2 from max_seq_len to save room for start and end tokens
    seq_pair_tuples = [ gen_source_target_pair() for in in 1:batch_size]
    seq_pair_batch  = convert_tuples_to_batch( seq_pair_tuples)

    return seq_pair_batch
end

# generate pair of source-target data. The target is the same as the source except if there is a five in the sequence
# all 5s are changed to 7s. If there is a 7 in the sequence all 7s are changed to 5s. If there is a 5 and a 7 
# both the target is an unaltered copy
function data_gen_contextual_pair( d_vocab, seqlen)
    x = [1 rand(4:d_vocab, 1, seqlen) 2]

    t = copy(x);
    i = x .== 5;
    j = x .== 7;
    if any(i) 
        if !any(j) 
            t[i] .= 7
        end
    elseif any(j)
        t[j] .= 5
    end

    return (x,t);
end

# convert a collection of source and target sequence tuples to a tuple of matrices
# each row of the matrix will have one source (or target sequence). The sequences will be padded with 3
function convert_tuples_to_batch( seq_pair_tuples )
   
   batch_length  = length(seq_pair_tuples)
   token_type    = eltype(eltype( seq_pair_tuples[1]))
   
   source_length = maximum( s->length(s[1]), seq_pair_tuples);
   source        = fill( token_type(3), ( batch_length, source_length))
   target_length = maximum( s->length(s[2]), seq_pair_tuples);
   target        = fill( token_type(3), ( batch_length, target_length))
   
   for (i, x) in enumerate( seq_pair_tuples)
      source[i,1:length(x[1])] = x[1];
      target[i,1:length(x[2])] = x[2];
   end

   return (source, target)
end

# Recursive zygote update method, this is the general recursion case:
function zyg_update!(opt, model, updates)
	# If this `model` node has no fields, then just return it
    if nfields(model) == 0
        return model
    end

	# If it does have fields, recurse into them:
    for field_idx in 1:nfields(model)
        zyg_update!(opt, getfield(model, field_idx), getfield(updates, field_idx))
    end

    # In the end, return the `model`
    return model
end
# If the `updates` is set to `Nothing`, then just return `model`; this means
# that there were no changes to be applied to this piece of the model.
zyg_update!(opt, model, updates::Nothing) = model

# If `model` is an `AbstractArray` and `updates` is too, then apply our Flux
# optimizer to the incoming gradients and apply them to the model!
function zyg_update!(opt, model::AbstractArray, updates::AbstractArray)
    # Sub off to Flux's ADAM optimizer
    Flux.Optimise.apply!(opt, model, updates)
    return model .-= updates
end