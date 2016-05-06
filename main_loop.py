import gc
from log import Log
import logging

logger = logging.getLogger(__name__)


class MainLoop(object):

    def __init__(self, model, datastream, extensions=None, log=None, 
                 static_input=dict(), testing=False):
        self.model = model
        logger.info('Compiling function')
        self.algorithm = self.model.algorithm
        self.test_algorithm = self.model.test_algorithm
        logger.info('Function compiled')
        self.datastream = datastream
        self.extensions = extensions
        self.testing = testing

        if log is None:
            log = Log()
        self.log = log

        self.status['training_started'] = False
        self.status['epoch_started'] = False
        self.output_map = {}
        self.outputs = self.model.outputs
        self.output_storage = {}
        self.static_input = static_input

        for i in range(len(self.model.outputs)):
            self.output_map[self.model.outputs[i].name] = i

    @property
    def status(self):
        return self.log.status
        
    #@profile
    def run(self):
        #logging.basicConfig()
        logger.info('Entered Main Loop')
        if not self.status['training_started']:
            for ext in self.extensions: # give each extension access
                ext.main_loop = self #    to the main loop
            self.run_extensions('before_training')
            self.status['training_started'] = True

        if self.status['iterations_done'] > 0:
            self.log.resume()
        
        try:
            while self.run_epoch():
                pass
        except TrainingFinish:
            self.log.current_row['training_finished'] = True

        self.run_extensions('after_training')

    #@profile
    def run_epoch(self):
        self.status['epoch_started'] = True
        try:
            self.epoch_iterator = (self.datastream.get_epoch_iterator(as_dict=True))
        except StopIteration:
            return False
        
        self.run_extensions('before_epoch')
        
        while self.run_iteration():
            pass

        self.status['epoch_started'] = False
        self.status['epochs_done'] += 1
        self.status['_epoch_ends'] += [self.status['iterations_done']]
        self.run_extensions('after_epoch')
        self.check_done()
        return True

    #@profile
    def run_iteration(self):
        try:
            batch = next(self.epoch_iterator)
        except StopIteration:
            return False
        
        self.run_extensions('before_batch', batch)
        ordered_batch = [(batch[v.name] if v.name in batch else self.static_input[v.name])
                         for v in self.model.inputs]

        self.outputs = self.algorithm(*ordered_batch)
        self.status['iterations_done'] += 1
        self.run_extensions('after_batch', batch)
        del batch # try to free memory (maybe)
        del ordered_batch
        #gc.collect()
        if self.testing:
            return False
        return True
    #@profile
    def run_extensions(self, method, *args):
        for extension in self.extensions:
            extension.notify(method, *args)


    def check_done(self):
        if self.log.current_row.get('training_finish_requested', False):
            raise TrainingFinish


class TrainingFinish(Exception):
    pass
