
"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) during training mode. In test mode values are scaled by 1-p (the retain rate)
 (http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). 

 NO one does it this way. They do it like Flux.Dropout. In particular. During 
 testing phase, nothing is done. During training, the values are either dropped out (set 
 to 0), or scaled by 1/(1-p). The ideas is that then the row and column averages will be 
 expected to be the same (after some are dropped out).

"""
mutable struct Dropout{F}
  p::F
  training::Bool
end

function Dropout(p)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p)}(p, true)
end


_dropout_kernel(y::T, p, scale) where {T} = y > p ? T(1/scale) : T(0)

function (a::Dropout)(x)
    # if not active (prob dropout = 0) return x unaltered
    a.p == 0 && return x;

    # if not in training mode
    !a.training && return x;

    # if training mode, then dropout (set input to 0) with probability p
    y = similar(x)
    Flux.rand!(y)
    y .= _dropout_kernel.(y, a.p, 1-a.p) # calc reciprocal prior to broadcast
  return x .* y
end

_testmode!(a::Dropout, test) = (a.training = !test)