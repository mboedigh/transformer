
# Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, 
# we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 
# “positional encodings” to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have
#  the same dimension dmodel as the embeddings, so that the two can be summed. 
struct PositionalEncoding
    d_maxpos::Integer
    d_model::Integer
    dropout::Flux.Dropout
    encodings

    # ...we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. 
    function PositionalEncoding(d_maxlen, d_model, p_drop) 
    # In this work, we use sine and cosine functions of different frequencies: PE(pos,2i)
    # where pos is the position (1..d_in <= d_maxlen) and i is the dimension from (1..2i=dim_model) 
    # PE(pos,2i) = sin(pos/10000^(2i/dmodel))
    # PE(pos,2i+1) = cos(pos/10000^(2i/dmodel)
    
    # The wavelengths form a geometric progression from 2π to (10000 · 2π) (so freq = 1..10000, also geometric)
        freq = exp10.(range(log10(1.0f0), log10(10000.0f0), length=d_model)); # create geometric progress for wavelength
        freq = 1f4 .^ ((1:d_model)./Float32(d_model));
        wavelength = 1 ./ freq;
    
        pe(pos,i) = i%2==0 ? sin(pos.*wavelength[i]) : cos(pos.*wavelength[i]) 
        
        # The positional encodings have the same dimension, dmodel, as the embeddings, so that the two can be summed. 
        encodings = Array{Float32}( undef, d_maxlen, d_model);
        for pos in 1:d_maxlen, i in 1:d_model
            encodings[pos, i] = pe(pos,i);
        end
    
        new(d_maxlen, d_model, Flux.Dropout(p_drop), encodings);
    end
end

PositionalEncoding(d_maxlen, d_model; p_drop = 0.1f0)  = PositionalEncoding(d_maxlen, d_model, p_drop)

# ...we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
(p::PositionalEncoding)(x) =  p.dropout( x + p.encodings[1:size(x,1),:] );

# batch version. 
# x is seqlen*batch x d_model matrix. sequences must all be of length seqlen
function (p::PositionalEncoding)(x::AbstractMatrix, seqlen::Int) 
n_s = size(x,1) // seqlen;
A = repeat( p.encodings[1:seqlen,:], Int(n_s), 1);
p.dropout( x .+ A );
end

