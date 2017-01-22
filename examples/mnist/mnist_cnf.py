from __future__ import division, print_function, absolute_import

import tensorflow as tf

from tefla.core.lr_policy import StepDecayPolicy
from tefla.da.standardizer import SamplewiseStandardizer
from tefla.utils import util

cnf = {
    'name': __name__.split('.')[-1],
    'classification': True,
    'iterator_type': 'parallel',  # parallel or queued
    'batch_size_train': 128,
    'batch_size_test': 128,
    'aug_params': {
        'zoom_range': (1 / 1.05, 1.05),
        'rotation_range': (-5, 5),
        'shear_range': (0, 0),
        'translation_range': (-2, 2),
        'do_flip': False,
        'allow_stretch': True,
    },
    'standardizer': SamplewiseStandardizer(clip=6),
    'num_epochs': 30,
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
