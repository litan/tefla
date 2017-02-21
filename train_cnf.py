from __future__ import division, print_function, absolute_import

import tensorflow as tf

from tefla.core.lr_policy import StepDecayPolicy
from tefla.utils import util

cnf = {
    'name': __name__.split('.')[-1],
    'classification': True,
    'iterator_type': 'queued',  # parallel or queued
    'batch_size_train': 16,
    'batch_size_test': 16,
    'l2_reg': 0.002,
    'aug_params': {
        'zoom_range': (1 / 1.05, 1.05),
        'rotation_range': (-5, 5),
        'shear_range': (0, 0),
        'translation_range': (-20, 20),
        'do_flip': True,
        'allow_stretch': True,
    },
    'num_epochs': 100,
    'summary_every': 5,
    'lr_policy': StepDecayPolicy(
        schedule={
            0: 0.001,
            15: 0.0001,
        }
    ),
    'optimizer': tf.train.AdamOptimizer(),
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
}
