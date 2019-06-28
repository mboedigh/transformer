
import Flux
import Flux.Chain

RepeatedLayer( layers::AbstractArray) = Flux.Chain( (layers...) )

applychain(::Tuple{}, x, xs...) = x
applychain(fs::Tuple, x, xs...) = applychain(Base.tail(fs), Base.first(fs)(x, xs...), xs...)
(c::Chain)(x,xs...) = applychain(c.layers, x, xs...)
 
#  Flux.children(c::RepeatedLayer) = c.layers
#  Flux.mapchildren(f, c::RepeatedLayer) = RepeatedLayer([f.(c.layers)...])
 
#  Base.getindex(c::RepeatedLayer, i::AbstractArray) = c.layers[i]
#  Base.getindex(c::RepeatedLayer, i::Integer) = c.layers[i]

