
import Flux

# "In addition to attention sub-layers, each of the layers in our encoder and decoder contains a 
#  fully connected feed-forward network, which is applied to each position separately and identically. 
#  This consists of two linear transformations with a ReLU activation in between."
# "The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048."
struct PositionwiseFeedForward
    w1::Linear
    w2::Linear
end

#  This consists of two linear transformations with a ReLU activation in between."
PositionwiseFeedForward( d_in, d_inner, d_out ) = return PositionwiseFeedForward( Linear(d_in, d_inner, Flux.relu), Linear(d_inner, d_out))

(ff::PositionwiseFeedForward)(x) = ff.w2(ff.w1(x));

@Flux.treelike PositionwiseFeedForward