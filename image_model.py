# ========= STD Libs  ============
from __future__ import division
import logging

# ========= Theano/npy ===========
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

# ========= Tools  ==============
from model import Model

# ========= Layers  ==============
import layers.activations as act
from layers.dropout import dropout
from layers.lstm2_layer import LSTMLayer
from layers.hidden_layer import HiddenLayer
from layers.read_layer import ReadLayer
from layers.write_layer import WriteLayer
from initialize import IsotropicGaussian

from lang_model import LanguageModel

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30))
np_rng = np.random.RandomState()

def zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)
def flatten_f(aList):
    t = []
    for i in aList:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten_f(i))
    return t


class ImageModel(Model):
    def __init__(self, bs, K, lang_N, steps, read_size, write_size, m, gen_dim, infer_dim, z_dim, l, seed=12345, channels=1, image_size=60*60):
        # K is the vocab size
        # lang_N is the (max) length of the sentence encoding
        # N is the number of times to run the model
        # m is the size of the langauge representation
        # l is the dimensions in the align function
        # image_size is the w*h of image (assumed square)
        self.use_gpu = True
        self.batch_size = bs
        self.gen_dim = gen_dim
        self.z_dim = z_dim
        self.m = m
        self.lang_N = lang_N
        self.steps = steps
        self.l = l
        self.read_size = read_size
        self.write_size = write_size
        self.infer_dim = infer_dim
        self.image_size = image_size

        self.language_model = LanguageModel(bs, K, lang_N, m)

        self.gen_in = HiddenLayer(input_size=m+z_dim, hidden_size=gen_dim*4,
                                  batch_size=bs, name='gen-lstm-in')
        self.gen_lstm = LSTMLayer(hidden_size=gen_dim, 
                                  activation=T.tanh, 
                                  batch_size=bs,
                                  dropout=0.0,
                                  name='gen-lstm')
        self.infer_in = HiddenLayer(input_size=2*self.read_size**2+self.gen_dim,
                                    hidden_size=infer_dim*4,
                                    batch_size=bs, name='infer-lstm-in')
        self.infer_lstm = LSTMLayer(hidden_size=infer_dim, 
                                    activation=T.tanh, 
                                    batch_size=bs,
                                    dropout=0.0,
                                    name='infer-lstm')

        self.reader = ReadLayer(batch_size=self.batch_size,
                                N=self.read_size,
                                channels=channels,
                                image_width=int(np.sqrt(self.image_size)),
                                image_height=int(np.sqrt(self.image_size)),
                                input_hidden_size=gen_dim,
                                name='Read')
        self.writer = WriteLayer(batch_size=self.batch_size,
                                 N=self.write_size,
                                 channels=channels,
                                 image_width=int(np.sqrt(self.image_size)),
                                 image_height=int(np.sqrt(self.image_size)),
                                 input_hidden_size=gen_dim,
                                 name='Write')
        self.random = RandomStreams(seed)

        # create W_mu, W_sigma, v, U, W, b
        init = IsotropicGaussian(0.01)
        u = init.init(np_rng, (self.m, self.l))
        self.U = theano.shared(value=u, name='U', borrow=True)
        v = init.init(np_rng, (self.l,))
        self.v = theano.shared(value=v, name='v', borrow=True)
        w = init.init(np_rng, (self.gen_dim, self.l))
        self.W = theano.shared(value=w, name='W', borrow=True)
        b = init.init(np_rng, (self.l,))
        self.b = theano.shared(value=b, name='b', borrow=True)


        w_mu = init.init(np_rng, (self.infer_dim, self.z_dim))
        self.W_mu_infer = theano.shared(value=w_mu, name='W_mu_infer', borrow=True)
        w_sigma = init.init(np_rng, (self.infer_dim, self.z_dim))
        self.W_sigma_infer = theano.shared(value=w_sigma, name='W_sigma_infer', borrow=True)

        w_mu = init.init(np_rng, (self.gen_dim, self.z_dim))
        self.W_mu_gen = theano.shared(value=w_mu, name='W_mu_gen', borrow=True)
        w_sigma = init.init(np_rng, (self.gen_dim, self.z_dim))
        self.W_sigma_gen = theano.shared(value=w_sigma, name='W_sigma_gen', borrow=True)


    def batched_dot(self, A, B):
        if self.use_gpu:
            return theano.sandbox.cuda.blas.batched_dot(A, B)
        else:
            return T.batched_dot(A,B)

    @property
    def params(self):
        return flatten_f([self.U, self.v, self.W, self.b, self.W_mu_infer, self.W_sigma_infer,
                self.W_mu_gen, self.W_sigma_gen] + self.language_model.params + \
        self.gen_in.params + self.gen_lstm.params + self.infer_in.params + self.infer_lstm.params + \
        self.reader.params + self.writer.params)

    def align(self, h_gen, h_lang, mask):
        # h_lang is N x batch x m
        # h_gen is batch x gen_dim
        # U is m x l
        # mask determines which elements of h_lang we care about
        # we want the result to be N x batch x l

        # using batched_dot this can be done
        # by making U to be 1 x m x l
        # and mkaing h_lang to be N x batch x m
        # and repeating U N times on axis 0
        U = self.U.reshape((1, self.m, self.l)).repeat(self.lang_N, axis=0)
        # align_lang is now N x batch x l
        align_lang = self.batched_dot(h_lang, U)

        # W is gen_dim x l
        # h_gen is batch x gen_dim
        # result is batch x l
        align_img = T.dot(h_gen, self.W)
        # use broadcasting to get a to be N x batch x l
        alpha = T.tanh(align_lang + align_img.dimshuffle('x',0,1) + self.b.dimshuffle('x','x',0))
        
        # v is l, a is N x batch x l
        # result will be N x batch
        alpha = T.exp(T.dot(alpha, self.v))
        
        # need to mask a before normalizing
        # so that the parts that are masked do
        # not affect the normalization
        mask = mask.transpose([1,0]) # make mask langN x batch_size
        alpha = T.switch(mask, alpha, zeros((self.lang_N, self.batch_size)))

        # normalize a by the sum of a along the N (axis=0)
        # creates a vector of length N
        alpha = alpha / T.sum(alpha, axis=0)

        # we now use alpha with h_lang to compute s_t
        # s_t is of size m because it is a constant
        # (alpha) * h_lang (m-vector)
        # we have alpha as N x batch
        # and h_lang as N x batch x m
        s = h_lang * alpha.reshape((self.lang_N, self.batch_size, 1))

        # sum along the N axis to give batch x m
        s = T.sum(s, axis=0)
        return s


    # use with partial to pass in first args
    # scan will pass the remaining args
    def step_train(self, rnd_in, kl, h_infer, c_infer, h_gen, c_gen, c, mu_gen, sigma_gen, h_lang, x, mask):
        # h_gen is a sequence
        # h_lang is a non-sequence (but it is used to calculate
        #     the align function each step)

        # eqs 10-13
        # compute "error image"
        x_hat = x-T.nnet.sigmoid(c)
        # read from both input (x) and error image
        r, _ = self.reader.run(x, h_gen)
        r_hat, _ = self.reader.run(x_hat, h_gen)
        # concatente the two read regions
        r = T.concatenate([r,r_hat], axis=1)
        
        # run the infer lstm on the read regions
        val = self.infer_in.run(T.concatenate([r, h_gen], axis=1))
        h_infer_t, c_infer_t = self.infer_lstm.run(val, h_infer, c_infer)
        
        # I don't believe we actually need to sample from Q
        # we just use it to minimze the loss so that it learns
        # good values for the infer-lstm
        # But we do need the mean and logsigma for KL
        mu_infer = T.dot(h_infer_t, self.W_mu_infer)
        sigma_infer = 0.5*T.dot(h_infer_t, self.W_sigma_infer)
        # generate a sample from these normal distributions
        z = mu_infer + T.exp(sigma_infer) * rnd_in
        
        # calculate kl-divergence between infer and gen normal distributions
        kl_t = kl + T.sum(-1 + ((mu_infer - mu_gen)**2 + T.exp(2*sigma_infer))/
                          (T.exp(2*sigma_gen)) - 2*sigma_infer + 2*sigma_gen)
        
        # generate a sample from the generative distribution
        z = mu_gen + T.exp(sigma_gen) * rnd_in

        # do the alignment (eq 2)
        # this is m-dimensions - each word is summed into 1 vector
        # to represent the whole sequence, so N x batch x m becomes batch x m
        s = self.align(h_gen, h_lang, mask)

        # run the LSTM (eq 3)
        # val is batch x m+z_dims
        val = self.gen_in.run(T.concatenate([z,s], axis=1))
        h_gen_t, c_gen_t = self.gen_lstm.run(val, h_gen, c_gen)

        mu_gen = T.tanh(T.dot(h_gen_t, self.W_mu_gen))
        sigma_gen = T.tanh(T.dot(h_gen_t, self.W_sigma_gen))

        # do the write (eq 4)
        c_update, _ = self.writer.run(h_gen_t)
        c_t = c + c_update

        return kl_t, h_infer_t, c_infer_t, h_gen_t, c_gen_t, c_t, mu_gen, sigma_gen
    

    def train(self, x, y, mask):
        # do language model on y
        h_lang = self.language_model.run(y)


        # do train recurrence
        h_infer, c_infer = self.infer_lstm.get_initial_hidden
        h_gen, c_gen = self.gen_lstm.get_initial_hidden
        c0 = theano.shared(-10*np.ones((1, self.image_size)).astype(theano.config.floatX))
        c0 = c0.repeat(self.batch_size, axis=0)

        rnd_in = rng.normal(size=(self.steps, self.batch_size, self.z_dim), 
                            avg=0.0, std=1.0, dtype=theano.config.floatX)

        # setup output
        outputs_info = [dict(initial=T.zeros(()), taps=[-1]), # kl
                        dict(initial=h_infer, taps=[-1]), # h_infer
                        dict(initial=c_infer, taps=[-1]), # c_infer
                        dict(initial=h_gen, taps=[-1]), # h_gen
                        dict(initial=c_gen, taps=[-1]), # c_gen
                        dict(initial=c0, taps=[-1]), # c
                        dict(initial=T.zeros((self.batch_size,self.z_dim)), taps=[-1]), # mu_gen
                        dict(initial=T.zeros((self.batch_size,self.z_dim)), taps=[-1])] # sigma_gen

        # do scan
        [kl, h_infer, c_infer, h_gen, c_gen, c, mu_gen, sigma_gen], _ = theano.scan(
                                                                         fn=self.step_train,
                                                                         sequences=rnd_in,
                                                                         outputs_info=outputs_info,
                                                                         non_sequences=[h_lang,x,mask],
                                                                         n_steps=self.steps)
                    

        # Get x-reconstruction-error (eq 5)
        x_recons = T.nnet.sigmoid(c[-1,:,:])
        log_recons = T.nnet.binary_crossentropy(x_recons, x).sum()
        
        # compute KL
        kl = 0.5*kl[-1]

        log_likelihood = kl + log_recons
        log_likelihood = log_likelihood.mean()
        kl = kl.mean()
        log_recons = log_recons.mean()
        return kl, log_recons, log_likelihood, c


    def generate_image(self, y, mask):
        # do language model on y
        h_lang = self.language_model.run(y)

        # do train recurrence
        h_gen, c_gen = self.gen_lstm.get_initial_hidden
        c0 = theano.shared(-10*np.ones((1, self.image_size)).astype(theano.config.floatX))
        c0 = c0.repeat(self.batch_size, axis=0)

        rnd_in = rng.normal(size=(self.steps, self.batch_size, self.z_dim), 
                            avg=0.0, std=1.0, dtype=theano.config.floatX)

        # setup output
        outputs_info = [dict(initial=h_gen, taps=[-1]), # h_gen
                        dict(initial=c_gen, taps=[-1]), # c_gen
                        dict(initial=c0, taps=[-1]), # c
                        dict(initial=T.zeros((self.batch_size,self.z_dim)), taps=[-1]), # mu_gen
                        dict(initial=T.zeros((self.batch_size,self.z_dim)), taps=[-1])] # sigma_gen

        # do scan
        [h_gen, c_gen, c, mu_gen, sigma_gen], _ = theano.scan(fn=self.step_gen,
                                                              sequences=rnd_in,
                                                              outputs_info=outputs_info,
                                                              non_sequences=[h_lang,mask],
                                                              n_steps=self.steps)
        c = T.nnet.sigmoid(c)

        return c[-1].reshape((1,self.batch_size,self.image_size))

    def step_gen(self, rnd_in, h_gen, c_gen, c, mu_gen, sigma_gen, h_lang, mask):
        # generate a sample from the generative distribution
        z = mu_gen + T.exp(sigma_gen) * rnd_in

        # do the alignment (eq 2)
        # this is m-dimensions - each word is summed into 1 vector
        # to represent the whole sequence, so N x batch x m becomes batch x m
        s = self.align(h_gen, h_lang, mask)

        # run the LSTM (eq 3)
        # val is batch x m+z_dims
        val = self.gen_in.run(T.concatenate([z,s], axis=1))
        h_gen_t, c_gen_t = self.gen_lstm.run(val, h_gen, c_gen)

        mu_gen = T.tanh(T.dot(h_gen_t, self.W_mu_gen))
        sigma_gen = T.tanh(T.dot(h_gen_t, self.W_sigma_gen))

        # do the write (eq 4)
        c_update, _ = self.writer.run(h_gen_t)
        c_t = c + c_update
            
        return h_gen_t, c_gen_t, c_t, mu_gen, sigma_gen


    def build_sample_function(self, y, mask):
        c = self.generate_image(y, mask)
        self.sample_sentences = theano.function([y, mask], [c])


    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
    
    @property
    def updates(self):
        return self._updates
