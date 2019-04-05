using Statistics

"""
    Diagonal(in::Integer)

Creates an element-wise linear transformation layer with learnable
vectors `α` and `β`:

    y = α.*x  .+ β

The input `x` must be a array where `size(x, 1) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end
Diagonal(in::Integer; initα = ones, initβ = zeros) =  Diagonal(Flux.param(initα(1,in)), Flux.param(initβ(1,in)))
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
 
# define a row based normalization to use with Flux.LayerNorm
function normalise(x,dims)
    u = mean(x, dims=dims);
    p = 1 ./ std(x,dims=dims);
    return (x .- u).*p
end
(a::LayerNorm)(x,dims=2) = a.diag(normalise(x,dims))

struct Sublayer{T}
    fn::T
    layernorm::LayerNorm
    dropout::Dropout;
end
Sublayer( f, d_in; p_drop = 0.1 ) = Sublayer( f, LayerNorm(d_in), Dropout(p_drop) )
Sublayer( f, d_in, p_drop = 0.1 ) = Sublayer( f, LayerNorm(d_in), Dropout(p_drop) )
@Flux.treelike Sublayer

# We employ a residual connection around each of the two sub-layers, followed by layer normalization. 
# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function 
# implemented by the sub-layer itself."
# We apply dropout to the output of each sub-layer, before it is added to the sub-layer input..."
# I take this to mean that x is the sub-layer input. Sublayer (without dashes) is the function
(s::Sublayer)(x, xs... ; p_drop = 0.1) = s.layernorm( x + s.dropout(s.fn(x, xs...)),2 )
