__precompile__(false)

# Based on the paper [Attention is all you need](http://arxiv.org/abs/1706.03762)
# and a reference implementation [Annotated Transfomer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
# a second reference implementation [https://github.com/chengchingwen/Transformers.jl]
# both reference implementations are good in that they seem to at least solve the toy problems I set up
# Many of the explanatory comments throughout my code are taken directly from the paper

module Transformers
using LinearAlgebra

include("Linears.jl")
include("Sublayer.jl")
include("Embedding.jl")
include("PositionalEncoding.jl")
include("PositionwiseFeedForward.jl")
include("Attention.jl")
include("RepeatedLayer.jl")
include("Encoder.jl")
include("Generator.jl")
include("TransformerLoss.jl")

export Sublayer
export Embedding
export PositionalEncoding
export PositionwiseFeedForward
export Linear
export MultiHeadedAttention
export Encoder
export Decoder
export LayerNorm
export RepeatedLayer
export Generator
export Transformer
export encode, decode, setdropoutmode!, predict, attention, embed, getmask
export transformer_loss

"""
    getmask( tokens::AbstractArray{T}, padding_idx = 3 ) 

return Int32 mask of 1s and 0s where 0s indicates padding tokens and 1s indicate meaningful tokens
return "nothing" if there is no padding (convention used in multi-headed attention)
tokens can be any type T and T must be comparable to padding_idx (default 3)
"""
function getmask( tokens::AbstractArray{T}, padding_idx = 3 ) where T
    is_padded = tokens .== T(padding_idx);
    findfirst(is_padded) == nothing && return nothing;
    
    mask = one(Int32) .- is_padded;  
end

"""
    Transformer

The Transformer model from "Attention is all you need"    
"""
struct Transformer
    source_embedding::Embedding
    positional_encoding::PositionalEncoding
    encoder_stack::RepeatedLayer
    decoder_stack::RepeatedLayer
    target_embedding::Embedding

    generator::Generator
end

"""
    Transformer() 

    returns a model as described in the paper "Attention is all you need", Vaswani et. al (2017).

**Keywords**
- max_seqlen = 1024   the maximum position in positional encoding. It must be larger the max seq length encountered
- d_vocab    = 13     the size of the shared word embedding matrix for both source and target sequences (includes room for 3 special tokens)
- d_model    = 512    the number of features in the embedding and throughout the model until the final projection
- n_heads    = 6      the number of attention heads used in Encoder and Decoder. Must divide model dimensions evenly
- n_layers   = 6      the number of Encoder (and Decoder) components in the encoder (and decoder) stacks
- p_drop     = 0.01f0 the drop probability used in Flux.Dropout after the embedding step and in each sublayer
"""
function Transformer(; max_seqlen = 1024, d_vocab=13, d_model = 512, n_heads = 8, n_layers = 6, p_drop = 0.1f0, init = Flux.glorot_uniform)

    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    W = Flux.param(init(d_vocab, d_model)); # need to create weights outside Embedding,to share them between embedding layers and pre-softmax transform
    source_embedding           = Embedding(W);   # my implementation did not do this because the reference implementation did not seem to do it

    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    #target_embedding    = Embedding(Flux.param(init(d_vocab, d_model))); # without sharing (and learning)
    target_embedding    = source_embedding; # with sharing

    positional_encoding = PositionalEncoding(max_seqlen, d_model; p_drop = p_drop);
    
    # "self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections"
    @assert d_model % n_heads == 0  "model dimensions (currently = $d_model) must be divisible by n_heads (currently = $n_heads)"
    d_attn              = Int32(ceil(d_model / n_heads))
    mha                 = MultiHeadedAttention(n_heads, d_model, d_attn; init=init);
    ff                  = PositionwiseFeedForward(d_model, d_model * 4, d_model); # "The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048."
    es = Array{Encoder}(undef, n_layers, 1)
    for i = 1:n_layers
        es[i] = Encoder(MultiHeadedAttention(n_heads, d_model, d_attn; init=init),
                        PositionwiseFeedForward(d_model, d_model * 4, d_model; initW=init); p_drop = p_drop);
    end
    encoder_stack       = RepeatedLayer(es)
    
    # The decoder is also composed of a stack of  N=6  identical layers
    ds = Array{Decoder}(undef, n_layers, 1)
    for i = 1:n_layers
        ds[i] = Decoder(MultiHeadedAttention(n_heads, d_model, d_attn; init=init),
        MultiHeadedAttention(n_heads, d_model, d_attn; init=init),
        PositionwiseFeedForward(d_model, d_model * 4, d_model; initW=init); p_drop = p_drop);
    end
    decoder_stack       = RepeatedLayer(ds)
    
    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    generator = Generator(d_model, d_vocab; init=init); # I am not sharing matrices because the math doesn't make sense to me. Seems like I would rather divide by embedding matrix than project with it

    return Transformer(source_embedding, positional_encoding, encoder_stack, decoder_stack, target_embedding, generator);
end

"""
    Transformer()(input, target)

return log probabilities for predicted output tokens. convert to predictions using `Flux.onecold(out')'`
Note: target sequence is masked during the run, so that token at target position i can only see tokens before target position i    

```jldoctest
julia> model = Transformer()
julia> input =  target = [1, 4, 5, 6, 2]; # simple copy task (one sequence as a vector)
julia> output = model( input, target ); # model is not yet fit, so these will be random
```

See also: [`embed`](@ref), [`encode`](@ref), [`decode`](@ref), [`predict`](@ref), [`generate`](@ref)
"""
function (t::Transformer)(source, target)
    memory = encode(t, source)
    mask   = getmask(target)
    out    = decode(t, target, memory, mask)
    t.generator(out);
end

"""
    embed( t::Transformer, x )

embed a tokens in d_model dimensional space with positional encoding
x can be a vector for a single sequence. If x is a matrix with each row representing a separate sequence of tokens

See also: [`Transformer`](@ref)
"""
function embed end

embed( t::Transformer, x )                = x |> t.source_embedding |> t.positional_encoding;  
function embed( t::Transformer, x::AbstractMatrix )  
    tlen = size(x,2); # length of target sequence
    t.source_embedding( vec(x')) |> x->t.positional_encoding(x, tlen);  
end

"""
encode(t::Transformer, x)

embeds and encodes x by passing x through embedding and all layers of the encoder_stack. 
x can be a single sequence or a matrix with rows representing separate sequences

See also: [`Transformer`](@ref), [`embed`](@ref)
"""
encode(t::Transformer, x::AbstractVector)               = embed(t,x) |> t.encoder_stack;
# x is a batch of sequences. Each row is one sequence and each column is one token of that sequence
function encode(t::Transformer, source::AbstractArray) 
    num_seqs, seq_len = size(source);
    t.source_embedding( vec(source'))    |> 
    x->t.positional_encoding(x,seq_len)  |> 
    x->t.encoder_stack(x,num_seqs);
end


"""
decode(t::Transformer, x, enc_output, mask)

embeds x and passes it through all layers of the decoder_stack. 
x can be a single sequence or a matrix with rows representing separate sequences
enc_output is output from the encoder_stack
mask is 1s where x has content and 0s where x is padded (i.e. padded to make seqeunces the same length)

See also: [`Transformer`](@ref), [`encode`](@ref), [`getmask`](@ref)
"""
decode(t::Transformer, x::AbstractVector, memory::AbstractMatrix, mask=nothing) = embed(t,x) |> x -> t.decoder_stack(x, memory, mask);
function decode(t::Transformer, target::AbstractMatrix, memory::AbstractMatrix, mask=nothing) 
    num_seqs, seq_len = size(target);
    t.target_embedding( vec(target')) |>  
    x->t.positional_encoding(x,seq_len)  |> 
    x->t.decoder_stack(x,memory, mask, num_seqs);
end


"""
    generate(t::Transformer, dec_output)

performs final projection and logsoftmax on decoder output

See also: [`Transformer`](@ref), [`encode`](@ref), [`getmask`](@ref)
"""
generate( t::Transformer, dec_output) = t.generator(dec_output)


"""
    setdropoutmode!(::Transformer, training::Bool = true)

sets the dropout mode of a Transformer model to true or false. dropout is used in each sublayer as well as the token embedding    
return the dropout mode prior to this call
"""
function setdropoutmode!(t::Transformer, training::Bool = true)
    curmode = t.positional_encoding.dropout.active;

    # set dropout in all layers to training
    t.positional_encoding.dropout.active = training;
    for l in t.encoder_stack.layers
        l.mha.dropout.active = training;
        l.ff.dropout.active = training;
    end

    # set dropout in all layers to training
    for l in t.decoder_stack.layers
        l.self_attn.dropout.active  = training;
        l.encoder_attn.dropout.active = training;
        l.ff.dropout.active = training;
    end
    return curmode
end

"""
   predict(::Transformer, input_sequence )

runs transformer model in evaluation mode to translate the given input sequence.    
output includes start and stop symbols (padding is indicated with 0s if needed)
"""
function predict(model::Transformer, datum; start_symbol=1, maxlen=nothing, stop_symbol=2)
    curmode = setdropoutmode!(model, false); # turn training off

    memory = encode(model, datum);
    ys     = Vector{eltype(datum)}(undef, 1);
    ys[1]  = start_symbol;
    maxlen == nothing && (maxlen = length(datum)*2 )
    for i in 2:maxlen
        out  = decode(model, ys[1:i - 1], memory, nothing ) # predict next word based previous decoding up to the current word, and memory from encoding
        yhat = model.generator(out[end,:]')
        word = Flux.onecold(yhat');
        push!(ys, word[end])
        word[end] == stop_symbol && break;
    end
    return ys

    setdropoutmode!(model, curmode);
end


function predict(model::Transformer, source, target)
    curmode =  setdropoutmode!(model, false);
    yhat = model(source, target);
    setdropoutmode!(model, curmode);
    ylabel = reshape( Flux.onecold(yhat'), size(target,2),:)'
    ylabel = [ones(eltype(ylabel), size(ylabel,1), 1 ) ylabel[:,1:end-1]];
    mask = getmask(target);
    mask == nothing || (ylabel .*= mask);
    ylabel
end

@Flux.treelike Transformer

function Base.show(io::IO, l::Transformer)
    d_model = size(l.source_embedding.W, 2);
    n_heads = l.encoder_stack.layers[1].mha.fn.n_heads;
    d_head =  Int(d_model/n_heads)
    print(io, "Transformer(d_model:$(d_model)) $(length(l.encoder_stack.layers)) layers and $(n_heads) heads (d_head:$(d_head)) in both encoder and decoder stacks")
end

end
