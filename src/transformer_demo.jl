#
using Statistics
using LinearAlgebra
using Flux, Zygote
using Transformers
# using Flux: gradient

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
    julia> run_transformer( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)

"""
function transformer_demo( ;max_seqlen=12, d_vocab=13, d_model=512, n_heads=8,
                            n_layers=2, p_drop = 0.01f0, n_batches = 20, batch_size = 30, n_epochs = 10)

    model   = Transformer(d_vocab=d_vocab, d_model=d_model, p_drop = p_drop, n_layers = n_layers);
    dataset = data_gen_copy_task( batch_size, d_vocab, max_seqlen, n_batches);

    run_transformer( model, dataset, n_epochs);

   # a = predict(model, 1:10 );
   # @assert all( a .== 1:10);  # this is not deterministic, and so is commented out. Do it with the model that transformer_demo() returns
end

"""
    data_gen_stutter_task(;batch_size=20, d_vocab=13, max_seqlen=12, n_batches = 30) 
Returns a dataset for stutter task. dataset is a collection of batches. max_seqlen is the size of the input sequence (before start and stop tokens are added)
this task tests batches of variable sequence length (each batch has uniform length sequences)
this task tests target sequence padding, since targets are of variable length
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_copy_task( ;n_batches=90, batch_size=30 );
    julia> run_transformer( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_stutter_task(;batch_size=20, d_vocab=13, max_seqlen=12, n_batches = 30) 
    seqlens = rand(4:max_seqlen, n_batches);
    [data_gen_batch( ()->data_gen_stutter_pair(d_vocab, seqlens[i]), batch_size ) for i in 1:n_batches]
end

"""
simplest task. target is an exact copy of the source
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_copy_task( ;n_batches=90, batch_size=30 );
    julia> run_transformer( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_copy_task(;batch_size=20, d_vocab=13, max_seqlen=12, n_batches = 30)    
    seqlens = rand(4:max_seqlen, n_batches);
    [data_gen_batch( ()->data_gen_copy_pair(d_vocab, seqlens[i]), batch_size)  for i in 1:n_batches]
end

"""
encode variable length sources (batches contain the sequences of the same length)
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_dyslexic_task( ;n_batches=90, batch_size=30 );
    julia> run_transformer( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_dyslexic_task(;batch_size=20, d_vocab=13, max_seqlen=12, n_batches = 30) 
    seqlens = rand(4:max_seqlen,n_batches);
    [data_gen_batch( ()->data_gen_dyslexic_pair(d_vocab, seqlens[i]), batch_size)  for i in 1:n_batches]
end

"""
this tests varied meaning of tokens depending on the presence of another token in the same sequence
encode variable length sources (batches contain the sequences of the same length)
    julia> model   = Transformer( ; transformer_hparams_tiny()... ) # note the ';' in call to Transformer (without it, there is a syntax error)
    julia> dataset = data_gen_contextual_task( ;n_batches=90, batch_size=30 );
    julia> run_transformer( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
"""
function data_gen_contextual_task(;batch_size=20, d_vocab=13, max_seqlen=12, n_batches = 30) 
    seqlens = rand(4:max_seqlen,n_batches);
    [data_gen_batch( ()->data_gen_contextual_pair(d_vocab, seqlens[i]), batch_size)  for i in 1:n_batches]
end

# train, or continue training, the transformer model over the dataset n_epoch (more) times.
# currentlyk resets the learning rate
function run_transformer( model, dataset, n_epochs = 10; stepnum=1)
    # optimizer
    warmup = 400;  # ramp up learning rate over 400 steps. Then decay as shown in learn_rate below
    opt = Flux.ADAM( learn_rate(stepnum, warmup), (0.9, 0.98) )
    ps = Flux.Params(Flux.params(model));
    for epoch in 1:n_epochs
        stepnum = transformer_epoch(model, dataset, opt, ps, epoch, stepnum); # this works in the script, but not on the command line
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
        :p_drop => 0.10f0,
        )
    end
    
# much smaller model but it will still eventually converged on the copy task
function transformer_demo_tiny()
    transformer_demo( ; n_epochs=30, n_batches=20, batch_size=30, transformer_hparams_tiny()... );
end

learn_rate(stepnum, warmup=4000, d_model=512) = (d_model.^-0.5f0) .* min.( stepnum.^-0.5f0 , stepnum .* warmup.^-1.5);

# process one epoch of data (sets of batches)
function transformer_epoch(model, dataset, opt, ps, epoch, stepnum )
    total_tokens = 0
    min_loss = Float32(Inf);
    for (batch_num, batch) in enumerate(dataset) 
        batch_start = time();
        print( "Epoch $epoch: ");
        opt.eta = learn_rate(stepnum,400);

        # train!(loss, ps, batch, opt) - I break it out in the next few lines below
        # ps = Params(ps);  # I do this no on the caller's side
        lbar = transformer_loss(model, batch...);  # call loss function to save the result
        gs = Flux.gradient( ()->lbar,ps);  
        Flux.Optimise.update!(opt, ps, gs);
        
        tokens = sum( batch[2] .!= 3) # sum non-padding target tokens
        total_tokens += tokens;
        rate = tokens/(time() - batch_start);

        lbar < min_loss && (min_loss = lbar);
        s = Base.Printf.@sprintf( "Batch: %d  learn_rate: %.5f tokens: %d token/s: %.1f batch_loss: %.2f min_batch_loss: %.2f",
        batch_num, opt.eta, tokens, rate, lbar, min_loss )
        println( s );
        stepnum += 1;
    end
    return stepnum;
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