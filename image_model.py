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
        # U is l x m
        # we want the result to be N x batch x l

        # using batched_dot this can be done
        # by making U to be 1 x m x l
        # and mkaing h_lang to be N x batch x m
        # and repeating U N times on axis 0
        U = self.U.reshape((1, self.m, self.l)).repeat(self.N, axis=0)
        # align_lang is now N x batch x l
        align_lang = self.batched_dot(h_lang, U)

        # W is l x gen_dim
        # h_gen is batch x gen_dim
        # result is batch x l
        align_img = T.dot(self.W, h_gen)
        # use broadcasting to get a to be N x batch x l
        alpha = T.tanh(align_lang + align_img + self.b)
        
        # v is l, a is N x batch x l
        # result will be N x batch
        alpha = T.exp(T.dot(alpha, self.v))
        
        # need to mask a before normalizing
        # so that the parts that are masked do
        # not affect the normalization
        alpha = T.switch(mask, alpha, zeros((self.N, self.l)))

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
    def step(self, train, h_infer, c_infer, stuff):
        # h_gen is a sequence
        # h_lang is a non-sequence (but it is used to calculate
        #     the align function each step)

        # draw samples from N(mu(h_gen), std(h_gen))
        # mu(h_gen) and std(h_gen) give a bach X z_dim vector
        mu_h_gen = T.tanh(T.dot(self.W_mu, h_gen))
        std_h_gen = T.exp(T.tanh(T.dot(self.W_sigma, h_gen)))

        # to draw samples from a normal, we want 1 sample for each
        # of the z_dims and 1 sample for each batch, so we want
        # a [1,1] output for the batch X z_dim vectors (eq 1)
        z = self.random.normal([1,1], avg=mu_h_gen, std=std_h_gen)

        # do the alignment (eq 2)
        # this is m-dimensions - each word is summed into 1 vector
        # to represent the whole sequence, so Nxm becomes m
        s = self.align(h_gen, h_lang)

        # run the LSTM (eq 3)
        val = self.gen_in.run(T.concatenate([z,s], axis=1))
        h_gen_t, c_gen_t = self.gen_lstm.run(val, h_gen, c_gen)

        # do the write (eq 4)
        c_t = c + self.writer.run(h_gen_t)
        
        if train:
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
            return h_infer_t, c_infer_t, c_t

    def train(self, stuff):
        # do scan

        # Get x-reconstruction-error (eq 5)
        x_recons = T.nnet.sigmoid(c[-1,:,:])
        
