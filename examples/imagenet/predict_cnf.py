from __future__ import division, print_function, absolute_import

import tensorflow as tf

from tefla.core.lr_policy import StepDecayPolicy
from tefla.da.standardizer import SamplewiseStandardizer
from tefla.utils import util

cnf = {
    'name': __name__.split('.')[-1],
    'iterator_type': 'queued',  # parallel or queued
    'batch_size_test': 128,
    # 'batch_size_train': 128,
    # 'classification': True,
    # 'aug_params': {
    #     'zoom_range': (1 / 1.05, 1.05),
    #     'rotation_range': (-5, 5),
    #     'shear_range': (0, 0),
    #     'translation_range': (-2, 2),
    #     'do_flip': False,
    #     'allow_stretch': True,
    # },
}
