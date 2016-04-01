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



def run():
    name = 'captioned-mnist'
    epochs = 500
    subdir = name + "-" + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    

    bs = 128
    # scale dataset to 224x224
    train_stream = DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, bs))
    valid_stream = DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, bs))
    test_stream  = DataStream.default_stream(data_test,  iteration_scheme=SequentialScheme(data_test.num_examples, bs))


    img_height, img_width = image_size

    
    x = T.matrix('features')
    y = T.matrix('captions')
    #    y.tag.test_value = np.random.rand(bs, 5, 20).astype('int16')
    
    K = 22
    lang_N = 10
    N = 32
    read_size = 8
    write_size = 8
    m = 256
    gen_dim = 300
    infer_dim = 300
    z_dim = 150
    l = 512

    model = ImageModel(bs, K, lang_N, N, read_size, write_size, m, gen_dim, infer_dim, z_dim, l)

    kl, log_recons, log_likelihood, c = model.train(x,y)
    kl.name = 'kl'
    log_recons.name = 'log_recons'
    log_likelihood.name = 'log_likelihood'
    c.name = 'c'

    model.set_outputs([kl, log_recons, log_likelihood, c])

    params = model.params

    from solvers.RMSProp import RMSProp as solver
    updates = solver(loss, params, lr=0.001)#, clipnorm=10.0)
    model.set_updates(updates)


    # ============= TRAIN =========
    plots = [['train_kl','valid_kl']]
    main_loop = MainLoop(net, train_stream,
                         [FinishAfter(epochs),
                          Track(variables=['kl','c'], prefix='train'),
                          TrackBest(variables=['kl'], prefix='train'),
                          DataStreamTrack(valid_stream, ['kl','c'], prefix='valid'),
                          Plot(name, plots, 'http://nameless-wave-6526.herokuapp.com/'),
                          SaveModel(subdir, name+'.model'),
                          TimeProfile(),
                          Printing()])
    main_loop.run()

if __name__ == '__main__':
#    theano.config.compute_test_value = 'warn'
#    theano.config.optimizer='None'#'fast_compile'
#    theano.config.exception_verbosity='high'
    run()
