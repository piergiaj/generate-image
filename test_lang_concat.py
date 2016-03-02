import numpy as np
import theano
import theano.tensor as T

N = 7
batch_size = 3
m = 2


def zeros(shape, dtype=theano.config.floatX):
    return theano.shared(np.zeros(shape).astype(dtype))

# y is N x batch x 1 (not one-hot encoded yet)
y = T.matrix()
y_rev = y[:,::-1]

hf = T.tensor3()
hb = T.tensor3()

b_indx = zeros((N, batch_size), int)
c = zeros((batch_size,), int)
for i in range(N):
    # if this part of y_rev is 0, ignore
    # else, get the current index
    indx = T.switch(T.neq(y_rev[:,i], 0), i, 0)
    # set b_indx to be the current indx if this is
    # a valid part of the string
    b_indx = T.set_subtensor(b_indx[c,T.arange(batch_size)], indx)
    
    # increment those that were used
    inc = T.switch(T.neq(y_rev[:,i], 0), 1, 0)
    c  = c + inc
    
h_b_aligned = hb[b_indx][:,T.arange(batch_size),T.arange(batch_size)]
h_lang = T.concatenate([hf, h_b_aligned], axis=2)

f = theano.function([y,hf,hb], h_lang, on_unused_input='warn')

y_in = [[4, 13, 45, 6, 0, 0, 0],
        [12, 4, 12, 0, 0, 0, 0],
        [3,  4,  5, 6, 7, 0, 0]]
hf =   [[[.8,.2], [.7,.3], [.5,.5], [.4,.6], [.4,.6], [.4,.6], [.4,.6]],
        [[.3,.7], [.5,.5], [.2,.8], [.2,.8], [.2,.8], [.2,.8], [.2,.8]],
        [[.9,.1], [.2,.8], [.4,.6], [.6,.4], [.7,.3], [.7,.3], [.7,.3]]]
hb =   [[[0,0],   [0,0],  [0,0],  [3,7],  [4,6],  [5,5],  [6,4]],
        [[0,0],   [0,0],  [0,0],  [0,0],  [6,4],  [7,3],  [8,2]],
        [[0,0],   [0,0],  [5,5],  [6,4],  [9,1], [12,-2],  [4,6]]]

# make them N(7) x batch(3) x m(2)
hf = np.asarray(hf).transpose([1,0,2])
hb = np.asarray(hb).transpose([1,0,2])

print f(y_in, hf, hb)
