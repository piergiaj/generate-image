from __future__ import division

import theano
from theano import tensor as T
import numpy as np

from hidden_layer import HiddenLayer
import activations as act


class WriteLayer(object):

    def __init__(self, batch_size, channels, N, image_width, image_height, input_hidden_size, use_dx_dy=False, name=''):
        """
        Write Layer from DRAW paper
        """

        self.batch_size = batch_size
        self.use_dx_dy = use_dx_dy
        self.N = N
        self.width = image_width
        self.height = image_height
        self.name = name
        self.input_hidden_size = input_hidden_size
        self.channels = channels
        self.test = False

        self.init_params()

    def init_params(self):
        self.transform_hidden = HiddenLayer(input_size=self.input_hidden_size,
                                            hidden_size=5+self.use_dx_dy,
                                            activation=act.Identity,
                                            name='Writer.Params.'+self.name)
        self.w_transform = HiddenLayer(input_size=self.input_hidden_size,
                                       hidden_size=self.channels*self.N*self.N,
                                       activation=act.Identity,
                                       name='Writer.Write.'+self.name)

    def batched_dot(self, A, B):
        C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])  
        return C.sum(axis=-2)

    def get_params(self, h):
        hidden = self.transform_hidden.run(h)
        
        gx = (hidden[:,0]+1)*0.5 * self.width
        gy = (hidden[:,1]+1)*0.5 * self.height
        s2 = T.exp(hidden[:,3]/2.0)
        g = T.exp(hidden[:,4]).dimshuffle(0,'x')
        if self.use_dx_dy:
            dx = (self.width-1.0) / (self.N-1.0) *  T.exp(hidden[:,2])
            dy = (self.height-1.0) / (self.N-1.0) *  T.exp(hidden[:,5])
        else:
            dx = dy = ((max(self.width,self.height)-1.0) / (self.N-1.0) * T.exp(hidden[:,2]))
        return gx,gy,dx,dy,s2,g

    def get_params_test(self, h):
        return h[:,0], h[:,1], h[:,2], h[:,5], h[:,3], h[:,4].dimshuffle(0,'x')

    def run(self, h):
        channels = self.channels#images.shape[1]
        if not self.test:
            gx,gy,dx,dy,s2,g = self.get_params(h)
        else:
            gx,gy,dx,dy,s2,g = self.get_params_test(h)

        w = self.w_transform.run(h)

        w = w.reshape((self.batch_size*self.channels, self.N, self.N))


        muX = gx.dimshuffle([0,'x']) + dx.dimshuffle([0,'x']) * (T.arange(self.N).astype(theano.config.floatX) - self.N/2 - 0.5)
        muY = gy.dimshuffle([0,'x']) + dy.dimshuffle([0,'x']) * (T.arange(self.N).astype(theano.config.floatX) - self.N/2 - 0.5)

        a = T.arange(self.width).astype(theano.config.floatX)
        b = T.arange(self.height).astype(theano.config.floatX)

        Fx = T.exp(-(a-muX.dimshuffle([0,1,'x']))**2 / 2. / s2.dimshuffle([0,'x','x'])**2)
        Fy = T.exp(-(b-muY.dimshuffle([0,1,'x']))**2 / 2. / s2.dimshuffle([0,'x','x'])**2)

        Fx = Fx / (Fx.sum(axis=-1).dimshuffle([0,1,'x']) + 1e-4)
        Fy = Fy / (Fy.sum(axis=-1).dimshuffle([0,1,'x']) + 1e-4)

        self.Fx = T.repeat(Fx, channels, axis=0)
        self.Fy = T.repeat(Fy, channels, axis=0)

        self.fint = self.batched_dot(self.Fy.transpose((0,2,1)), w)

        self.fim = self.batched_dot(self.fint, self.Fx).reshape((self.batch_size, self.channels*self.width*self.height))

        return 1./g * self.fim, (gx, gy, dx, dy, self.fint)

    @property
    def params(self):
        return [param for param in [self.transform_hidden.params]+[self.w_transform.params]]

    @params.setter
    def params(self, params):
        self.transform_hidden.params = params[:len(params)/2]
        self.w_transform.params = params[len(params)/2:]
