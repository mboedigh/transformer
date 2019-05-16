
import Flux

struct RepeatedLayer{T<:AbstractArray}
    layers::T
    RepeatedLayer(xs) = new{typeof(xs)}(xs)
end

# make repeatd layers manually to to initialize params separately 
# example:
# es = Array{Encoder}(undef, n_layers,1)
# for i = 1:n_layers
#     es[i] = Encoder( MultiHeadedAttention( n_heads, d_model, d_attn), 
#                      PositionwiseFeedForward(d_model, d_model*4, d_model ); p_drop = p_drop );
# end    
# encoder_stack       = RepeatedLayer(es)

        

function (e::RepeatedLayer)(x)
   for layer in e.layers
       x = layer(x)
   end
   return x
end

function (e::RepeatedLayer)(x, xs...)
    for layer in e.layers
        x = layer(x, xs...)
    end
    return x
 end
 
 Flux.children(c::RepeatedLayer) = c.layers
 Flux.mapchildren(f, c::RepeatedLayer) = RepeatedLayer(f.(c.layers)...)
 
 Base.getindex(c::RepeatedLayer, i::AbstractArray) = c.layers[i]
 Base.getindex(c::RepeatedLayer, i::Integer) = c.layers[i]

