
import Flux

struct RepeatedLayer{T<:AbstractArray}
    layers::T
    RepeatedLayer(xs) = new{typeof(xs)}(xs)
end
RepeatedLayer(layer, n) = return RepeatedLayer([deepcopy(layer) for i in 1:n]);
    
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

