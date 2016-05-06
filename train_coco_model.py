# ========= STD Libs  ============
from __future__ import division
from collections import OrderedDict
import os
import shutil
import sys
import logging
#import ipdb
import cPickle
import argparse
import time
sys.setrecursionlimit(100000000)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Theano/npy ===========
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import fuel
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

# ========= Tools  ==============
from main_loop import MainLoop
from extensions import Printing, FinishAfter, TrackBest, Track
from extensions.timing import TimeProfile
from extensions.plot import PlotLocal, Plot
from extensions.data_monitor import DataStreamTrack
from extensions.save_model import SaveModel, SaveBestModel


from image_model import ImageModel
from coco_dataset import MSCoco
from sample_coco_sentences import SampleSentences
from lr_ext import DropLearningRate

def run():
    name = 'coco-nopeep'
    epochs = 200
    subdir = name + "-" + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    

    bs = 200
    data_train = MSCoco(dataset='train', num=82611, bs=bs)
    data_valid = MSCoco(dataset='val', num=4989, bs=bs)

    train_stream = DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, bs))
    valid_stream = DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, bs))


    img_height, img_width = (32,32)

    
    x = T.matrix('features')
    #x.tag.test_value = np.random.rand(bs, 3*32*32).astype('float32')
    y = T.lmatrix('captions')
    #y.tag.test_value = np.random.rand(bs, 57).astype(int)
    mask = T.lmatrix('mask')
    #mask.tag.test_value = np.ones((bs,57)).astype(int)

    K = 25323
    lang_N = 57
    N = 32
    read_size = 9
    write_size = 9
    m = 256
    gen_dim = 550
    infer_dim = 550
    z_dim = 275
    l = 512

    model = ImageModel(bs, K, lang_N, N, read_size, write_size, m, gen_dim, infer_dim, z_dim, l, image_size=32*32, channels=3, cinit=0.0)
    model._inputs = [x,y,mask]

    kl, log_recons, log_likelihood, c = model.train(x,y,mask)
    kl.name = 'kl'
    log_recons.name = 'log_recons'
    log_likelihood.name = 'log_likelihood'
    c.name = 'c'

    model._outputs = [kl, log_recons, log_likelihood, c]

    params = model.params

    #from solvers.RMSProp import RMSProp as solver
    lr = theano.shared(np.asarray(0.001).astype(theano.config.floatX))
    #updates = solver(log_likelihood, params, lr=lr, clipnorm=10.0)#0.001)
    #lr = 0.001
    grads = T.grad(log_likelihood, params)
    his = []
    for p in params:
        pz = p.get_value()*0
        his.append(theano.shared(pz))

    threshold = 10.0
    decay = 0.9
    updates = OrderedDict()

    for p, ph, g in zip(params, his, grads):
        l2_norm = T.sqrt(T.sqr(g).sum())
        m = T.switch(l2_norm < threshold, 1, threshold/l2_norm)
        grad = m*g
        
        ph_n = decay * ph + (1-decay)*grad**2
        updates[ph] = ph_n
        updates[p] = p-(lr/T.sqrt(ph_n+1e-6))*grad
    
    model._updates = updates

    logger.info('Compiling sample function')
    model.build_sample_function(y, mask)
    logger.info('Compiled sample function')

    # ============= TRAIN =========
    plots = [['train_kl','valid_kl'],
             ['train_log_recons','valid_log_recons'],
             ['train_log_likelihood','valid_log_likelihood']]
    main_loop = MainLoop(model, train_stream,
                         [FinishAfter(epochs),
                          Track(variables=['kl','log_recons','log_likelihood'], prefix='train'),
                          #TrackBest(variables=['kl'], prefix='train'),
                          DataStreamTrack(valid_stream, ['kl','log_recons','log_likelihood'], prefix='valid'),
                          SampleSentences(subdir, bs, 32, 32),
                          DropLearningRate(lr, 25, 0.0001),
                          Plot(name, plots, 'http://nameless-wave-6526.herokuapp.com/'),
                          SaveModel(subdir, name+'.model'),
                          TimeProfile(),
                          Printing()])
    main_loop.run()

if __name__ == '__main__':
#    theano.config.compute_test_value = 'warn'
 #   theano.config.optimizer='fast_compile'
#    theano.config.exception_verbosity='high'
    run()

