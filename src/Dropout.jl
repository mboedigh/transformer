
"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) during training mode. In test mode values are scaled by 1-p (the retain rate)
 (http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

Does nothing to the input once in [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  training::Bool
end

function Dropout(p)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p)}(p, true)
end

_dropout_kernel(y::T, p) where {T} = y > p ? T(1) : T(0)

function (a::Dropout)(x)
    # if not active (prob dropout = 0) return x unaltered
    a.p == 0 && return x;

    # if not in training mode, then scale by probability of retaining 
    if (!a.training) 
        return x.*(1-a.p);
    end

    # if training mode, then dropout (set input to 0) with probability p
    y = similar(x)
    Flux.rand!(y)
    y .= _dropout_kernel.(y, a.p)
  return x .* y
end

_testmode!(a::Dropout, test) = (a.training = !test)