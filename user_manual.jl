## with different size source and target sequence
source = [1,2,4,4,5,6,7,8 ];
target = [1,2,2,4,5,5,7,7,8,3,3,3 ]; # put 3 at end because I build mask based on padding idx
mask   = [1,1,1,1,1,1,1,1,1,0,0,0];

model = Transformer( n_layers=2; init=(n,m)->init_repeats(n,m,7), p_drop = 0.0f0 )
enc  = encode(model,source);
dec  = decode(model,target,enc,getmask(target));
yhat = model.generator(dec);
lhat = Flux.onecold(yhat')'

transformer_loss( model, source, target)
q = Transformers.smooth_label(target[2:end], 13);
logp = yhat[1:end-1,:];
Transformers.kld_loss(q,logp,mask[1:end-1])

# compare output to Transformer.jl
m = get_model( init=(m,n)->init_repeats(m,n,7))
x_mask = getmask([[ source];[source]]); 
t_mask = getmask( [[target];[target]]);    t_mask[:,end,:] .= 0; # manually mask last token    
x = [source source] ;
t = [target target] ;
src = m.embedding(x);
enc = m.encoder(src);
mask = getmask( x_mask, t_mask );
trg = m.embedding(t);
dec = m.decoder(trg, enc, mask);

lab = onehot(vocab, t);
label = smooth(lab)[:, 2:end, :];
logkldivergence(label, dec[:, 1:end-1, :], t_mask[:, 1:end-1, :])


## Overview of The full process
#  generate a test dataset
#  generate a new untrained model
#  traing the model
#  evaluate the model

## configure a model
model = Transformer( ; transformer_hparams_tiny()... ); 
# a couple of prebuilt sets of hyper params are available. Here I create a tiny model that fits these problems will
# default parameters (model = Transformer()) create a model used in the paper

## create testing datasets
# A dataset is a collection of batched data
# each batch is a 2-tuple of arrays. The first array is a set of sequences (one per row) in the source language and 
# the second array is a set of sequences (again one per row) in the target language. 
# Each position in the sequence is an Integer token representing some word or syllable or other 
# semantic bit of a language. The arrays are padded with 3s 
# to allows for variable length sequences. Currently, only variable length targets are supported. 
# there are 4 built-in dataset generators to illustrate simple problems and examine the transformer
dataset = data_gen_stutter_task( batch_size=32, d_vocab=13, seqlen=10, n_batches=100);
# e.g. 
# batch = dataset[1];   # first batch of data
# source,target = batch; # sequences (one per row) in source and target language.
# see also
# data_gen_stutter_task(...)
# data_gen_dyslexic_task(...)
# data_gen_contextual_task(...)
#
# you may also easily create your own custom function. 
# * make a function that returns one matching pair of sequences (e.g. data_gen_copy_pair(...) )
# * make a two-line function that returns a dataset of batched data.  (e.g. data_gen_copy_task(...))
# 
#

# train the model
train_transformer!( model, dataset, 30); # run for 30 epochs using tiny model (took my tests < 15 epochs)
# this will run through the dataset 30 times 
# there is a lower level function that this one calls: transformer_epoch that will run through the dataset just once
# see train_transformer! for a simple example of how to set that up


# evaulate the model on a batch
curmode = setdropoutmode!(model, false); # turn off dropout n_layers
testset = data_gen_stutter_task( batch_size=32, d_vocab=13, seqlen=10, 1);
batch = testset[1]; # if testset is an array of batches or batch = testset if it is a singleton
loss = Transformers.transformer_loss( model, batch... )
setdropoutmode!(model, curmode); # change back to whatever mode model was in

# see the model predictcions
ypred = predict(model, batch[1])

# if it is not good enough, just keep training (turn on dropout first)
setdropoutmode!(model, true);
train_transformer!( model, dataset, 30);



 