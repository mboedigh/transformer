using Statistics

struct Sublayer{T}
    fn::T
    layernorm::Flux.LayerNorm
    dropout::Flux.Dropout;
end
Sublayer( f, d_in; p_drop = 0.1 ) = Sublayer( f, Flux.LayerNorm(d_in), Flux.Dropout(p_drop) )
Sublayer( f, d_in, p_drop = 0.1 ) = Sublayer( f, Flux.LayerNorm(d_in), Flux.Dropout(p_drop) )
@Flux.treelike Sublayer

# We employ a residual connection around each of the two sub-layers, followed by layer normalization. 
# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function 
# implemented by the sub-layer itself."
# We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized"

# I take this to mean that x is the sub-layer input. Sublayer (without dashes) is the function
# (s::Sublayer)(x, xs... ) = s.layernorm( x + s.dropout(s.fn(x, xs...)) )

# this is how it is done in the annotated transformer. It is not like the text implied though...
(s::Sublayer)(x, xs... ) = x + s.dropout( s.layernorm( s.fn(x, xs...)) )