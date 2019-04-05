import Base.copy

# Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, 
# we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 
# “positional encodings” to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have
#  the same dimension dmodel as the embeddings, so that the two can be summed. 
struct PositionalEncoding
    d_in
    d_model
    dropout::Dropout
    encodings

    # ...we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. 
    function PositionalEncoding(d_maxlen, d_model, p_drop) 
    # In this work, we use sine and cosine functions of different frequencies: PE(pos,2i)
    # where pos is the position (1..d_in) and i is the dimension from (1..2i=dim_model) 
    # PE(pos,2i) = sin(pos/10000^(2i/dmodel))
    # PE(pos,2i+1) = cos(pos/10000^(2i/dmodel)
        @assert d_model % 2 == 0
        len = convert(Int, d_model/2)
    
    # The wavelengths form a geometric progression from 2π to (10000 · 2π) (so freq = 1..10000, also geometric)
        freq = exp10.(range(log10(1), log10(10000), length=len)); # create geometric progress for wavelength
        wavelength = 1 ./ freq;
    
    # The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. 
        encodings = Array{Float32}( undef, d_maxlen, d_model);
        for i in 1:d_maxlen, j in 1:len
            encodings[i, j*2-1] = sin(i.*wavelength[j])
            encodings[i, j*2] = cos(i.*wavelength[j])
        end
    
        new(d_maxlen, d_model, Dropout(p_drop), encodings);
    
    end
end

PositionalEncoding(d_maxlen, d_model; p_drop = 0.1f0)  = PositionalEncoding(d_maxlen, d_model, p_drop)

function (p::PositionalEncoding)(x) 
    # ...we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
    p.dropout( x + p.encodings[1:size(x,1),:] );
end

