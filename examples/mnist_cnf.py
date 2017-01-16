from __future__ import division, print_function, absolute_import

import tensorflow as tf

from tefla.core.lr_policy import StepDecayPolicy
from tefla.utils import util
from tefla.da.standardizer import SamplewiseStandardizer

cnf = {
    'name': __name__.split('.')[-1],
    'batch_size_train': 128,
    'batch_size_test': 128,
    'classification': True,
    'standardizer': SamplewiseStandardizer(clip=6),
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    'aug_params': {
        'zoom_range': (1 / 1.05, 1.05),
        'rotation_range': (-5, 5),
        'shear_range': (0, 0),
        'translation_range': (-2, 2),
        'do_flip': False,
        'allow_stretch': True,
    },
    'num_epochs': 30,
    'lr_policy': StepDecayPolicy(
        schedule={
            0: 0.001,
            15: 0.0001,
        }
    ),
    'optimizer': tf.train.AdamOptimizer()
}
