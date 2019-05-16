using Statistics
using Flux

"""
    Diagonal(in::Integer)

Creates an feature-wise linear transformation layer with learnable
vectors `α` and `β` for each feature

    y = α.*x  .+ β

The input `x` must be a array where `size(x, 1) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end
Diagonal(in::Integer; initα = Flux.ones, initβ = Flux.zeros) =  Diagonal(Flux.param(initα(1,in)), Flux.param(initβ(1,in)))
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
 
# define a row based normalization to use with my version of LayerNorm (row-based)
function normalise(x,dims)
    return (x .- mean(x, dims=dims) )./ (std(x,dims=dims) .+ 1f-6)
end
(a::LayerNorm)(x,dims=2) = a.diag(normalise(x,dims))

# We employ a residual connection around each of the two sub-layers, followed by layer normalization. 
# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function 
# implemented by the sub-layer itself."
struct Sublayer{T}
    fn::T
    layernorm::Transformers.LayerNorm
    dropout::Flux.Dropout;
end
Sublayer( f, d_in; p_drop = 0.1 ) = Sublayer( f, Transformers.LayerNorm(d_in), Flux.Dropout(p_drop) )
Sublayer( f, d_in, p_drop = 0.1 ) = Sublayer( f, Transformers.LayerNorm(d_in), Flux.Dropout(p_drop) )
@Flux.treelike Sublayer


# We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized"
# I take this to mean that x is the sub-layer input. Sublayer (without dashes) is the function
# This is as written in the text and in Transformers.jl
(s::Sublayer)(x, xs... ) = s.layernorm( x + s.dropout(s.fn(x, xs...)),2 )

# this is how it is done in the annotated transformer. It is not like the text implied though...
# (s::Sublayer)(x, xs... ) = x + s.dropout( s.layernorm( s.fn(x, xs...), 2) )