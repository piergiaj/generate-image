from __future__ import division

import logging
import theano
import theano.tensor as T

import numpy as np

from main_loop import MainLoop
from model import Model
from extensions import Extension

class DropLearningRate(Extension):

    def __init__(self, lr, epochs, new_rate, **kwargs):
        self.lr = lr
        self.epochs = epochs
        self.new_rate = new_rate
        kwargs.setdefault('after_epoch', True)
        super(DropLearningRate, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        log = self.main_loop.log
        if log.status['epochs_done'] > self.epochs:
            self.epochs = 100000 # make it so large it never happens again
            self.lr.set_value(self.new_rate)
            print 'lr is now', self.new_rate
