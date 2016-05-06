from uuid import uuid4
from numbers import Integral
from collections import defaultdict

import six

class Log(defaultdict):

    def __init__(self, uuid=None):
        defaultdict.__init__(self, dict)
        self.status = {}
        if uuid is None:
            self.uuid = uuid4()
            self.status.update({
                'iterations_done': 0,
                'epochs_done': 0,
                '_epoch_ends':[],
                'resumed_from': None
            })
        else:
            self.uuid = uuid

    @property
    def h_uuid(self):
        return self.uuid.hex

    def resume(self):
        old_uuid = self.h_uuid
        old_status = dict(self.status)
        self.status['resumed_from'] = old_uuid

    def _check_time(self, time):
        if not isinstance(time, Integral) or time < 0:
            raise ValueError('Time is negative')

    @property
    def current_row(self):
        return self[self.status['iterations_done']]

    @property
    def previous_row(self):
        return self[self.status['iterations_done']-1]

    @property
    def last_epoch_row(self):
        return self[self.status['_epoch_ends'][-1]]


    def __reduce__(self):
        cons, args, _, _, items = super(Log, self).__reduce__()
        return cons, (), self.__dict__, _, items

    def __getitem__(self, time):
        self._check_time(time)
        return super(Log, self).__getitem__(time)

    def __setitem__(self, time, val):
        self._check_time(time)
        return super(Log, self).__setitem__(time, val)
