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
from layer.hidden_layer import HiddenLayer
from layers.read_layer import ReadLayer
from layers.write_layer import WriteLayer

def zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)


class ImageModel(Model):
    def __init__(self, bs, m, gen_dim, z_dim, l, seed=12345):
        # m is the size of the langauge representation
        # l is the dimensions in the align function
        self.bs = bs
        self.gen_dim = gen_dim
        self.z_dim = z_dim
        self.m = m

        self.gen_in = HiddenLayer(input_size=m+z_dim, hidden_size=gen_dim*4,
                                  batch_size=bs, name='gen-lstm-in')
        self.gen_lstm = LSTMLayer(hidden_size=gen_dim, 
                                  activation=T.tanh, 
                                  batch_size=bs,
                                  dropout=0.0,
                                  name='gen-lstm')
        self.random = RandomStreams(seed)

        # create W_mu, W_sigma, v, U, W, b

    def align(self, h_gen, h_lang, mask):
        # h_lang is N x batch x m
        # h_gen is batch x gen_dim
        # U is l x m
        # mask determines which elements of h_lang we care about
        # we want the result to be N x batch x l

        # using batched_dot this can be done
        # by making U to be 1 x m x l
        # and mkaing h_lang to be N x batch x m
        # and repeating U N times on axis 0
        U = self.U.reshape((1, self.m, self.l)).repeat(self.N, axis=0)
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
        alpha = T.switch(mask, alpha, zeros((self.N, self.batch_size)))

        # normalize a by the sum of a along the N (axis=0)
        # creates a vector of length N
        alpha = alpha / T.sum(alpha, axis=0)

        # we now use alpha with h_lang to compute s_t
        # s_t is of size m because it is a constant
        # (alpha) * h_lang (m-vector)
        # we have alpha as N x batch
        # and h_lang as N x batch x m
        s = h_lang * alpha.reshape((self.N, self.batch_size, 1))

        # sum along the N axis to give batch x m
        s = T.sum(s, axis=0)
        return s


    # use with partial to pass in first args
    # scan will pass the remaining args
    def step_train(self, rnd_in, kl, h_infer, c_infer, h_gen, c_gen, c, mu_gen, sigma_gen, h_lang):
        # h_gen is a sequence
        # h_lang is a non-sequence (but it is used to calculate
        #     the align function each step)

        # eqs 10-13
        # compute "error image"
        x_hat = x-T.nnet.sigmoid(c)
        # read from both input (x) and error image
        r = self.reader.run(x, h_gen)
        r_hat = self.reader.run(x_hat, h_gen)
        # concatente the two read regions
        r = T.concatenate([r,r_hat], axis=1)
        
        # run the infer lstm on the read regions
        val = self.infer_in.run(T.concatenate([r, h_gen], axis=1))
        h_infer_t, c_infer_t = self.infer_lstm.run(val, h_infer, c_infer)
        
        # I don't believe we actually need to sample from Q
        # we just use it to minimze the loss so that it learns
        # good values for the infer-lstm
        # But we do need the mean and logsigma for KL
        mu_infer = T.dot(self.W_mu_infer, h_infer_t)
        sigma_infer = 0.5*T.dot(self.W_sigma_infer, h_infer_t)
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
        s = self.align(h_gen, h_lang)

        # run the LSTM (eq 3)
        # val is batch x m+z_dims
        val = self.gen_in.run(T.concatenate([z,s], axis=1))
        h_gen_t, c_gen_t = self.gen_lstm.run(val, h_gen, c_gen)

        mu_gen = T.tanh(self.W_mu_gen, h_gen_t)
        sigma_gen = T.tanh(self.W_sigma_gen, h_gen_t)

        # do the write (eq 4)
        c_t = c + self.writer.run(h_gen_t)

        return kl_t, h_infer_t, c_infer_t, h_gen_t, c_gen_t, c_t, mu_gen, sigma_gen
    
    def step_gen(self, rnd_in, h_gen c_gen, mu_gen, sigma_gen, c, h_lang):
        # generate a sample from the generative distribution
        z = mu_gen + T.exp(sigma_gen) * rnd_in

        # do the alignment (eq 2)
        # this is m-dimensions - each word is summed into 1 vector
        # to represent the whole sequence, so N x batch x m becomes batch x m
        s = self.align(h_gen, h_lang)

        # run the LSTM (eq 3)
        # val is batch x m+z_dims
        val = self.gen_in.run(T.concatenate([z,s], axis=1))
        h_gen_t, c_gen_t = self.gen_lstm.run(val, h_gen, c_gen)

        mu_gen = T.tanh(self.W_mu_gen, h_gen_t)
        sigma_gen = T.tanh(self.W_sigma_gen, h_gen_t)

        # do the write (eq 4)
        c_t = c + self.writer.run(h_gen_t)
            
        return h_gen_t, c_gen_t, mu_gen, sigma_gen, c_t

    def train(self, x, y):
        # do language model on y

        # setup output
        outputs_info = [dict(initial=, taps=[-1]), # kl
                        dict(initial=, taps=[-1]), # h_infer
                        dict(initial=, taps=[-1]), # c_infer
                        dict(initial=, taps=[-1]), # h_gen
                        dict(initial=, taps=[-1]), # c_gen
                        dict(initial=, taps=[-1]), # c
                        dict(initial=, taps=[-1]), # mu_gen
                        dict(initial=, taps=[-1])], # sigma_gen

        # do scan
        theano.scan(fn=self.step_train,
                    sequences=rnd_in,
                    outputs_info=outputs_info,
                    non_sequences=h_lang,
                    n_steps=self.steps)
    def step_train(self, rnd_in, kl, h_infer, c_infer, h_gen, c_gen, c, mu_gen, sigma_gen, h_lang):

                    

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
