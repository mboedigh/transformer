

struct Encoder
    mha::Sublayer{MultiHeadedAttention};
    ff::Sublayer{PositionwiseFeedForward};
    norm::LayerNorm;
end

function Encoder( mha::MultiHeadedAttention, ff::PositionwiseFeedForward; p_drop = 0.1f0)
    n = size(mha.W.W,2);
    Encoder( Sublayer(mha,n,p_drop), Sublayer(ff,n,p_drop), LayerNorm(n))
end

# layernorm at this step is from "The Annotated Transformer, but not the paper"
(en::Encoder)(x) = return en.mha(x,x,x) |> en.ff |> en.norm

@Flux.treelike Encoder

# Decoder is made of self-attn, src-attn, and feed forward (defined below)
struct Decoder
    self_attn::Sublayer{MultiHeadedAttention};
    encoder_attn::Sublayer{MultiHeadedAttention};
    ff::Sublayer{PositionwiseFeedForward};
    norm::LayerNorm;
end
function Decoder( self::MultiHeadedAttention, memory::MultiHeadedAttention, ff::PositionwiseFeedForward; p_drop = 0.1f0) 
    n = size(self.W.W,2);
    Decoder( Sublayer(self,n, p_drop), Sublayer(memory,n, p_drop), Sublayer(ff,n, p_drop), LayerNorm(n))
end
@Flux.treelike Decoder

# "Follow Figure 1 (right) for connections."
# In "encoder-decoder attention" layers, the queries come from the previous decoder layer, 
# and the memory keys and values come from the output of the encoder. 
#  We employ a residual connection around each of the two sub-layers ...
function (en::Decoder)(x, memory) 
    return  en.self_attn( x,x,x,true)                |> 
            x -> en.encoder_attn( x, memory, memory) |> 
            en.ff |> en.norm   # layernorm at this step is from "The Annotated Transformer, but not the paper"
end

function Base.show(io::IO, l::Encoder)
    print(io, "Encoder($(l.mha.fn.n_heads) heads)" )
end                                    

