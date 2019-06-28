using Statistics
using Flux

"""
    Diagonal(in::Integer)

Creates an feature-wise linear transformation layer with learnable
vectors `α` and `β` for each feature

    y = α.*x  .+ β

The input `x` must be a array where `size(x, 2) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end
Diagonal(in::Integer; initα = Flux.ones, initβ = Flux.zeros) =  Diagonal(Flux.param(initα(1,in)), Flux.param(initβ(1, in)))
@Flux.treelike Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α .*x .+ β
end

struct LayerNorm{T}
    diag::Diagonal{T}
end
LayerNorm(h::Integer) = LayerNorm(Diagonal(h))
@Flux.treelike LayerNorm

# very similar to Flux.normalise, but with a stabilization factor added
function rownormalize(x)
    μ′ = mean(x, dims = 2);
    ϵ  = eps(maximum(μ′));
    # ϵ = zero(eltype(x));
    σ′ = std(x, dims = 2, mean = μ′, corrected=false) .+ ϵ;
    return (x .- μ′) ./ σ′
end
 
(a::LayerNorm)(x,dims=2) = a.diag(rownormalize(x))

"""
    Sublayer
  
We employ a residual connection around each of the two sub-layers, followed by layer normalization. 
That is, the output of each sub-layer is "LayerNorm(x + Sublayer(x))", where "Sublayer(x)" is the function 
implemented by the sub-layer itself.
We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized"
I take this to mean that x is the sub-layer input. Sublayer (without dashes) is the function
"""
struct Sublayer{T}
    fn::T
    layernorm::Transformers.LayerNorm
    dropout::Flux.Dropout;
end

"""
    Sublayer( f, d_model; p_drop = 0.1 )    

Create a Residual Connection Sublayer with function f with output dimensions d_model and drop probability pdrop
"""
Sublayer( f, d_model; p_drop::Real = 0.1f0 ) = Sublayer( f, Transformers.LayerNorm(d_model), Flux.Dropout(p_drop) )
@Flux.treelike Sublayer

"""
    (s::Sublayer)(x, xs... )

execute the residual sublayer by calling the configured Sublayer with input x (and optional inputs xs...)
"""
(s::Sublayer)(x, xs... ) = s.layernorm( x + s.dropout(s.fn(x, xs...)),2 )
