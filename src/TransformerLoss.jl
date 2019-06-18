

# label is sequence of tokens (up to max_seq_len)
function smooth_label( label::AbstractVector, d_vocab )
    # each column represents a possible token in d_vocab
    # each row is an element of the sequence

    smoothing = Float64(1e-6);
    cool_value   = Float32( smoothing/d_vocab);
    y_smooth = fill( cool_value, size(label,1), d_vocab); 
    warm_value = Float32( 1 ).- convert(Float32, smoothing);
    for (i,j) = enumerate(label)
        y_smooth[i,j] = warm_value
    end
    y_smooth;
end

smooth_label(label::AbstractMatrix, d_vocab) = smooth_label(vec(label'),d_vocab)

```
yhat is logsoftmax predictions (each row is a token)
y_smooth is a onehot or a smoothed "onewarm" matrix using the function smooth_label
mask, if provided, is a vector of weights (e.g. 1s and 0s with 0s representing tokens that should not be considered) 
```
function ce_loss(yhat, y_smooth, mask=nothing)
    n_tokens, d_vocab = size(yhat);
    crossentropy = sum( yhat .* y_smooth, dims=2);
    if (mask != nothing)
        n_tokens = sum(mask);
        crossentropy = crossentropy .* mask; 
    end
    -sum(crossentropy)/n_tokens;
end

function kld_loss( q, logp, mask=nothing )
    n_tokens, d_vocab = size(logp);
    kld = (q .* (log.(q .+ eps(q[1])) .- logp)) 
    if (mask != nothing)
        n_tokens = sum(mask);
        kld = kld .* mask;
        t = copy(kld)
    end
    sum(kld)/n_tokens;
end

```
Transformer.loss

average cross_entropy over all unmasked tokens (masked tokens are excluded)
yhat is logsoftmax predictions (each row is a token)
labels is a vector of correct tokens. These will be smoothed to produce a "onewarm" matrix using the function smooth_label
mask, if provided, is a vector of weights (e.g. 1s and 0s with 0s representing tokens that should not be considered) 
```
function transformer_loss( model, source::AbstractVector, target::AbstractVector)
    logp   = model(source, target)[1:end-1,:];  # prediction i is for target[i+1], so stop before last predicted token
    d_vocab = size(logp,2);

    q = Transformers.smooth_label( target[2:end], d_vocab);
    Transformers.kld_loss(q, logp, getmask(target[2:end]) ); 
end

```
calculate sum of mean loss per token for each sequence in batch
sequences are masked in positions where the token uses the padding character (3). 
```
function transformer_loss(model, source::AbstractMatrix, target::AbstractMatrix)

    d_vocab  = size(model.source_embedding.W,1);
    target_seq_len = size(target,2);

    mask     = getmask(target);  # find and mask padding characters in target sequence
    memory   = encode( model, source);
    out      = decode( model, target, memory, mask);
    yhat     = model.generator( out );
    
    y_smooth = Transformers.smooth_label(vec(target'), d_vocab);    # make one long run on sentence 
    
    q = view(y_smooth, 2:size(y_smooth,1), :);
    qq = (q .* (log.(q .+ eps(q[1])) .- yhat[1:end-1,:]));
    loss = sum( q .* (log.(q .+ eps(q[1])) .- yhat[1:end-1,:]), dims=2); # sum across all possible vocab for each position in sequence
    
    # mask start of every next sequence in target (start_symbol), because we only use target[2:end] for every target. 
    shift_mask = ones(eltype(loss.data), size(target));
    if (mask != nothing)              # if there is a target mask, then mask all those positions as well
        shift_mask .*= mask;
    end
    shift_mask[:,1] .= 0;             # mask first token in each sequence since it isn't compared
    n = sum(shift_mask, dims=2);        # unmasked tokens per sequence
    shift_mask ./= n;             # weight for each token is 1/n, where n is the number of unmasked tokens in the sequence
    shift_mask = vec(shift_mask')[2:end]; 
    loss  = loss .* shift_mask;
    
    sum(loss);        
end

