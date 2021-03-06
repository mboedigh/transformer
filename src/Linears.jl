__precompile__(false)

import Flux

struct Linear{F,S,T}
    W::S
    b::T
    σ::F
end
  
Linear(W, b) = Linear(W, b, identity)
  
function Linear(in::Integer, out::Integer, σ = identity; initW = Flux.glorot_uniform, initb = Flux.zeros)
    return Linear(Flux.param(initW(in, out)), Flux.param(initb(1, out)), σ)
end

@Flux.treelike Linear

(a::Linear)(x::AbstractArray) =   a.σ.(x*a.W .+ a.b)

