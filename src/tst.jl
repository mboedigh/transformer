using  Revise
using  LinearAlgebra    
using  Statistics
import Flux

using Transformers

struct Pages
    A::AbstractArray
end
Base.iterate(r::Pages) =  size(r.A,1)>0 ? (r.A[1,:,:],2) : nothing
Base.iterate(r::Pages, state) =  state <= size(r.A,1) ? (r.A[state,:,:], state+1) : nothing
Base.length(r::Pages) = size(r.A,1)
Base.eltype(r::Pages) = eltype(r::A)

struct Rows
    A::AbstractArray
end
Base.iterate(r::Rows) =  size(r.A,1)>0 ? (r.A[1,:],2) : nothing
Base.iterate(r::Rows, state) =  state <= size(r.A,1) ? (r.A[state,:], state+1) : nothing
Base.length(r::Rows) = size(r.A,1)
Base.eltype(r::Rows) = eltype(r::A)

struct Cols
    A::AbstractArray
end
Base.iterate(r::Cols) =  size(r.A,2)>0 ? (r.A[:,1],2) : nothing
Base.iterate(r::Cols, state) =  state <= size(r.A,2) ? (r.A[:,state], state+1) : nothing
Base.length(r::Cols) = size(r.A,2)
Base.eltype(r::Cols) = eltype(r::A)

struct LabelSmoothing
    # "Implement label smoothing."
    criterion
    padding_idx
    confidence
    size
    true_dist
end

using Distances

function LabelSmoothing( size, padding_idx, smoothing=0.0, true_dist = nothing )
        return LabelSmoothing( Distances.KLDivergence(), padding_idx, 1.0-smoothing, size, true_dist)
end
        
function (self::LabelSmoothing)(x)
        @assert size(x,1) == self.size
        true_dist = similar(x)
        fill!( true_dist, self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
end