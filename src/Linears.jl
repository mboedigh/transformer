import Flux

struct Linear{F,S,T}
    W::S
    b::T
    σ::F
end
  
Linear(W, b) = Linear(W, b, identity)
  
function Linear(in::Integer, out::Integer, σ = identity;
    initW = Flux.glorot_uniform, initb = zeros)
    return Linear(param(initW(in, out)), param(initb(out)), σ)
end

@Flux.treelike Linear

function (a::Linear)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(x*W .+ b')
end