from __future__ import division, print_function, absolute_import

import logging
import math
import pprint

logger = logging.getLogger('tefla')


class InitialLrMixin(object):
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr


class NoBatchUpdateMixin(object):
    def batch_update(self, learning_rate, iter_idx, n_iter_per_epoch):
        return learning_rate


class NoEpochUpdateMixin(object):
    def epoch_update(self, learning_rate, training_history):
        return learning_rate


class NoDecayPolicy(InitialLrMixin, NoBatchUpdateMixin, NoEpochUpdateMixin):
    def __str__(self):
        return 'NoDecayPolicy(rate=%s)' % str(self.initial_lr)

    def __repr__(self):
        return str(self)


class StepDecayPolicy(NoBatchUpdateMixin):
    def __init__(self, schedule):
        self.schedule = schedule
        self.initial_lr = self.schedule[0]

    def resume_lr(self, start_epoch, n_iter_per_epoch, resume_lr):
        if resume_lr is not None:
            resume_lr = float(resume_lr)
            self._update_schedule(resume_lr, start_epoch)
            return resume_lr
        else:
            return self._lr_for_epoch(start_epoch)

    def _update_schedule(self, resume_lr, epoch):
        skeys = sorted(self.schedule.keys())
        next_step = next((x for x in skeys if x > epoch), None)
        if next_step is None:
            prev_step_index = len(skeys) - 1
        else:
            prev_step_index = skeys.index(next_step) - 1
        prev_lr = self.schedule[skeys[prev_step_index]]
        lr_ratio = resume_lr / prev_lr
        for idx in range(prev_step_index, len(skeys)):
            curr_lr = self.schedule[skeys[idx]]
            self.schedule[skeys[idx]] = curr_lr * lr_ratio
        logger.info('Updated step decay schedule: %s' % pprint.pformat(self.schedule))

    def _lr_for_epoch(self, epoch):
        skeys = sorted(self.schedule.keys())
        next_step = next((x for x in skeys if x > epoch), 1)
        prev_step_index = skeys.index(next_step) - 1
        return self.schedule[skeys[prev_step_index]]

    def epoch_update(self, learning_rate, training_history):
        epoch_info = training_history[-1]
        epoch = epoch_info['epoch']
        new_learning_rate = learning_rate
        if epoch in self.schedule.keys():
            new_learning_rate = self.schedule[epoch]
        return new_learning_rate

    def __str__(self):
        return 'StepDecayPolicy(schedule=%s)' % pprint.pformat(self.schedule)

    def __repr__(self):
        return str(self)


class PolyDecayPolicy(InitialLrMixin, NoEpochUpdateMixin):
    def __init__(self, initial_lr, power=10.0, max_epoch=500):
        self.power = power
        self.max_epoch = max_epoch
        super(PolyDecayPolicy, self).__init__(initial_lr)

    def resume_lr(self, start_epoch, n_iter_per_epoch, resume_lr):
        if resume_lr is not None:
            self.initial_lr = float(resume_lr)
        return self.batch_update(None, (start_epoch - 1) * n_iter_per_epoch, n_iter_per_epoch)

    def batch_update(self, learning_rate, iter_idx, n_iter_per_epoch):
        new_learning_rate = self.initial_lr * math.pow(1 - iter_idx / float(self.max_epoch * n_iter_per_epoch),
                                                       self.power)
        return new_learning_rate

    def __str__(self):
        return 'PolyDecayPolicy(initial rate=%f, power=%f, max_epoch=%d)' % (self.initial_lr, self.power,
                                                                             self.max_epoch)

    def __repr__(self):
        return str(self)
