module Transformers
import Flux

include("Dropout.jl")
include("Sublayer.jl")
include("Embedding.jl")
include("PositionalEncoding.jl")
include("PositionwiseFeedForward.jl")
include("Attention.jl")
include("RepeatedLayer.jl")
include("Encoder.jl")
include("Generator.jl")

export Dropout
export Sublayer
export Embedding
export PositionalEncoding
export PositionwiseFeedForward
export Attention
export MultiHeadedAttention
export Encoder
export Decoder
export RepeatedLayer
export Generator
export Transformer
export encode, decode

struct Transformer
    source_embedding
    positional_encoding
    encoder_stack
    decoder_stack
    target_embedding

    generator
end

function Transformer( max_seq_len, d_vocab, d_model=512; n_heads=8, n_layers=6, p_drop = 0.1 ) 
    
    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    W = Flux.param(rand(Float32, d_vocab, d_model)); # need to create weights outside Embedding,to share them between embedding layers and pre-softmax transform
    embedding           = Embedding(W);   

    positional_encoding = PositionalEncoding( max_seq_len, d_model; p_drop = p_drop);

    # "self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections"
    @assert d_model % n_heads == 0  "model dimensions (currently = $d_model) must be divisible by n_heads (currently = $n_heads)"
    d_attn              = Int32( ceil( d_model/n_heads))
    mha                 = MultiHeadedAttention( n_heads, d_model, d_attn);
    ff                  = PositionwiseFeedForward(d_model, d_model*4, d_model ); # "The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048."
    encoder_stack       = RepeatedLayer( Encoder(mha, ff; p_drop = p_drop), n_layers);
    
    # The decoder is also composed of a stack of  N=6  identical layers
    c = deepcopy;
    decoder_self_attn   = MultiHeadedAttention( n_heads, d_model, d_attn);
    decoder_stack       = RepeatedLayer( Decoder(decoder_self_attn, c(mha), c(ff); p_drop = p_drop), n_layers);

    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    target_embedding    = Embedding(W); # share the same weight matrix between the two embedding layers and the pre-softmax linear transformation

    # In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    generator = Generator(d_model, d_vocab);

    return Transformer( embedding, positional_encoding, encoder_stack, decoder_stack, target_embedding, generator);
end

function encode(t::Transformer, x) 
    return x |> t.source_embedding |> t.positional_encoding |> t.encoder_stack;
end

function decode( t::Transformer, x, memory) 
    return x |> t.target_embedding |> t.positional_encoding |> x -> t.decoder_stack(x, memory)
end

function (t::Transformer)(source, target) 
    memory = encode(t,source)
    out    = decode(t,target,memory)
    yhat   = t.generator(out)
    return yhat
end

function setmode( t::Transformer, training = false)

    curmode = t.positional_encoding.dropout.training;
    
    # set dropout in all layers to training 
    t.positional_encoding.dropout.training = training;
    for l in t.encoder_stack.layers
        l.mha.dropout.training = training;
        l.ff.dropout.training = training;
    end

    # set dropout in all layers to training 
    for l in t.decoder_stack.layers
        l.self_attn.dropout.training = training;
        l.encoder_attn.dropout.training = training;
        l.ff.dropout.training = training;
    end
    return curmode
end

function predict( model::Transformer, datum, start_symbol = 1 )
    curmode = setmode(model, false); # turn traning off

    memory = encode(model, datum);
    ys = similar(datum);
    ys[1] = start_symbol;
    for i in 2:length(datum)
        out  = decode( model, ys[1:i-1], memory ) # predict next word based decoding of current word, and memory from encoding
        yhat = model.generator( out[end,:]' )
        word = Flux.onecold(yhat);
        ys[i] =  word[1] # set next word. TODO: check this is right word -- I think it is always a single element, but be sure
    end
    return ys

    setmode(model,curmode);
end

@Flux.treelike Transformer

function Base.show(io::IO, l::Transformer)
    print(io, "Transformer(d_model:$(size(l.source_embedding.W,2)); encoding: $(length(l.encoder_stack.layers)) layers and $(length(l.encoder_stack.layers[1].mha.fn.heads)) heads; decoding: $(length(l.encoder_stack.layers)) layers)" )
end



end