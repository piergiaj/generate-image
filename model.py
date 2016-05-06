import theano

class Model(object):

    def __init__(self):
        pass

    @property
    def algorithm(self):
        if not hasattr(self, '_algorithm'):
            self._algorithm = theano.function(inputs=self.inputs,
                                              outputs=self.outputs,
                                              updates=self.updates,on_unused_input='warn')
        return self._algorithm
    
    @property
    def test_algorithm(self):
        if not hasattr(self, '_talgorithm'):
            self._talgorithm = theano.function(inputs=self.inputs,
                                               outputs=self.outputs)
        return self._talgorithm

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def outputs(self):
        raise NotImplementedError
    @property
    def updates(self):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError
