# Based on the paper [Attention is all you need](http://arxiv.org/abs/1706.03762)
# and a reference implementation [Annotated Transfomer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and the
# Many of the explanatory comments throughout my code are taken directly from the paper

__precompile__(false)
module Transformers
# using Flux
using LinearAlgebra

include("Sublayer.jl")
include("Embedding.jl")
include("PositionalEncoding.jl")
include("PositionwiseFeedForward.jl")
include("Linears.jl")
include("Attention.jl")
include("RepeatedLayer.jl")
include("Encoder.jl")
include("Generator.jl")

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
export encode, decode, setdropoutmode, predict, attention

# return mask of 1s and -Inf for positions with content or padding (-Inf)
function getmask( tokens::AbstractArray{T} ) where T
    mask = zeros(Float32, size(tokens));
    mask[ tokens .== 3 ] .= Float32(-1e9);  # the way Transformers.jl does it, -Inf causes softmax to return NaN
    mask;
end

struct Transformer
    source_embedding
    positional_encoding
    encoder_stack
    decoder_stack
    target_embedding

    generator
end

# Transformer
#    Transformer() returns a model as described in the paper "Attention is all you need"
# max_seqlen is in regard to max positional encoding. It only needs to be larger than the input sequence length
function Transformer(; max_seqlen = 1024, d_vocab=11, d_model = 512, n_heads = 8, n_layers = 6, p_drop = 0.1f0)

    init = Flux.glorot_uniform;
    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    W = Flux.param(init(d_vocab, d_model)); # need to create weights outside Embedding,to share them between embedding layers and pre-softmax transform
    source_embedding           = Embedding(W);   # my implementation did not do this because the reference implementation did not seem to do it

    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    #target_embedding    = Embedding(Flux.param(init(d_vocab, d_model))); # without sharing
    println("shared embedding")
    target_embedding    = source_embedding; # with sharing

    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    generator = Generator(d_model, d_vocab); # I am not sharing matrices because the math doesn't make sense to me. Seems like I would rather divide by embedding matrix than project with it

    positional_encoding = PositionalEncoding(max_seqlen, d_model; p_drop = p_drop);

    # "self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections"
    @assert d_model % n_heads == 0  "model dimensions (currently = $d_model) must be divisible by n_heads (currently = $n_heads)"
    d_attn              = Int32(ceil(d_model / n_heads))
    mha                 = MultiHeadedAttention(n_heads, d_model, d_attn);
    ff                  = PositionwiseFeedForward(d_model, d_model * 4, d_model); # "The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048."
    es = Array{Encoder}(undef, n_layers, 1)
    for i = 1:n_layers
        es[i] = Encoder(MultiHeadedAttention(n_heads, d_model, d_attn),
                         PositionwiseFeedForward(d_model, d_model * 4, d_model); p_drop = p_drop);
    end
    encoder_stack       = RepeatedLayer(es)

    # The decoder is also composed of a stack of  N=6  identical layers
    ds = Array{Decoder}(undef, n_layers, 1)
    for i = 1:n_layers
        ds[i] = Decoder(MultiHeadedAttention(n_heads, d_model, d_attn),
                        MultiHeadedAttention(n_heads, d_model, d_attn),
                        PositionwiseFeedForward(d_model, d_model * 4, d_model); p_drop = p_drop);
    end
    decoder_stack       = RepeatedLayer(ds)

    return Transformer(source_embedding, positional_encoding, encoder_stack, decoder_stack, target_embedding, generator);
end

function encode(t::Transformer, x)
    return x |> t.source_embedding |> t.positional_encoding |> t.encoder_stack;
end

function decode(t::Transformer, x, memory, mask)
    return x |> t.target_embedding |> t.positional_encoding |> x -> t.decoder_stack(x, memory,mask)
end

function (t::Transformer)(source, target)
    memory = encode(t, source)
    mask = getmask(target)
    out    = decode(t, target, memory, mask)
    yhat   = t.generator(out)
    return yhat
end

function setdropoutmode(t::Transformer, training::Bool = true)
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

function predict(model::Transformer, datum; start_symbol=1, maxlen=nothing, stop_symbol=2)
    curmode = setdropoutmode(model, false); # turn training off

    memory = encode(model, datum);
    ys     = Vector{eltype(datum)}(undef, 1);
    ys[1]  = start_symbol;
    maxlen == nothing && (maxlen = length(datum)*2 )
    for i in 2:maxlen
        out  = decode(model, ys[1:i - 1], memory, nothing ) # predict next word based decoding of current word, and memory from encoding
        yhat = model.generator(out[end,:]')
        word = Flux.onecold(yhat');
        push!(ys, word)
        word == stop_symbol && break;
    end
    return ys

    setdropoutmode(model, curmode);
end

@Flux.treelike Transformer

function Base.show(io::IO, l::Transformer)
    print(io, "Transformer(d_model:$(size(l.source_embedding.W, 2))); $(length(l.encoder_stack.layers)) layers and $(l.encoder_stack.layers[1].mha.fn.n_heads) heads in both encoder and decoder stacks")
end

end
