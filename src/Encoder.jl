#Encoder Layer
#    Each layer has 2 sublayers
#    One sublayer is for multi-headed self-attention
#    The other sublayer is for a two-layer feedforward network

struct Encoder
    mha::Sublayer{MultiHeadedAttention};
    ff::Sublayer{PositionwiseFeedForward};
end

function Encoder( mha::MultiHeadedAttention, ff::PositionwiseFeedForward; p_drop = 0.1f0)
    n = size(mha.W.W,2);
    Encoder( Sublayer(mha,n,p_drop=p_drop), Sublayer(ff,n,p_drop=p_drop))
end

(en::Encoder)(x) = en.mha(x,x,x) |> en.ff # as in the paper, and the Transformers.jl
(en::Encoder)(x, num_seqs) = en.mha(x,x,x,num_seqs) |> en.ff
@Flux.treelike Encoder

# Decoder is made of self-attn, src-attn, and feed forward (defined below)
struct Decoder
    self_attn::Sublayer{MultiHeadedAttention};
    encoder_attn::Sublayer{MultiHeadedAttention};
    ff::Sublayer{PositionwiseFeedForward};
end
function Decoder( self::MultiHeadedAttention, memory::MultiHeadedAttention, ff::PositionwiseFeedForward; p_drop = 0.1f0)
    n = size(self.W.W,2);
    Decoder( Sublayer(self,n, p_drop=p_drop), Sublayer(memory,n, p_drop=p_drop), Sublayer(ff,n, p_drop=p_drop))
end
@Flux.treelike Decoder

# "Follow Figure 1 (right) for connections."
# In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
# and the memory keys and values come from the output of the encoder.
#  We employ a residual connection around each of the two sub-layers ...
# x is the target sequence (whole sequence for training, or characters decoded so far for eval)
# mask is 1 in positions that are part of the sequence and 0 for positions that are padding
# memory is output from encoder stack
function (de::Decoder)(x, memory, mask)
    return  de.self_attn( x,x,x, nothing, true)            |>
            x -> de.encoder_attn( x, memory, memory, mask) |>
            de.ff 
end

# batch version
function (de::Decoder)(x, memory, mask, num_seqs)
    return  de.self_attn( x,x,x, num_seqs, nothing, true)            |>
            x -> de.encoder_attn( x, memory, memory, num_seqs, mask) |>
            de.ff  
end

function Base.show(io::IO, l::Encoder)
    print(io, "Encoder($(l.mha.fn.n_heads) heads)" )
end

function Base.show(io::IO, l::Decoder)
    print(io, "Decoder($(l.self_attn.fn.n_heads) heads)" )
end