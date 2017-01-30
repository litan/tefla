from __future__ import division, print_function, absolute_import

import functools
import logging

from tefla import convert
from tefla.core.layer_arg_ops import make_args
from tefla.da import data
from tefla.da import iterator
from tefla.da.standardizer import NoOpStandardizer

logger = logging.getLogger('tefla')


def create_training_iters(cnf, data_set, crop_size, epoch, parallel):
    standardizer = cnf.get('standardizer', NoOpStandardizer())
    balancing = 'balance_ratio' in cnf
    if parallel:
        if balancing:
            training_iterator_maker = iterator.BalancingDAIterator
        else:
            training_iterator_maker = iterator.ParallelDAIterator
        validation_iterator_maker = iterator.ParallelDAIterator
        logger.info('Using parallel training iterator')
    else:
        if balancing:
            training_iterator_maker = iterator.BalancingQueuedDAIterator
        else:
            training_iterator_maker = iterator.QueuedDAIterator
        validation_iterator_maker = iterator.ParallelDAIterator
        # validation_iterator_maker = iterator.QueuedDAIterator
        logger.info('Using queued training iterator')

    preprocessor = None
    if balancing:
        balancing_args = make_args(
            balance_weights=data_set.balance_weights(),
            final_balance_weights=cnf['final_balance_weights'],
            balance_ratio=cnf['balance_ratio'],
            balance_epoch_count=epoch - 1,
        )
    else:
        balancing_args = {}

    training_iterator = training_iterator_maker(
        batch_size=cnf['batch_size_train'],
        shuffle=True,
        preprocessor=preprocessor,
        crop_size=crop_size,
        is_training=True,
        aug_params=cnf.get('aug_params', data.no_augmentation_params),
        standardizer=standardizer,
        fill_mode='constant',
        # save_to_dir=da_training_preview_dir,
        **balancing_args
    )

    validation_iterator = validation_iterator_maker(
        batch_size=cnf['batch_size_test'],
        shuffle=True,
        preprocessor=preprocessor,
        crop_size=crop_size,
        is_training=False,
        standardizer=standardizer,
        fill_mode='constant'
    )

    return training_iterator, validation_iterator


def convert_preprocessor(im_size):
    return functools.partial(convert.convert, crop_size=im_size)


def create_prediction_iter(cnf, crop_size, preprocessor=None, sync=False):
    standardizer = cnf.get('standardizer', NoOpStandardizer())
    if sync:
        prediction_iterator_maker = iterator.DAIterator
    else:
        prediction_iterator_maker = iterator.ParallelDAIterator

    prediction_iterator = prediction_iterator_maker(
        batch_size=cnf['batch_size_test'],
        shuffle=False,
        preprocessor=preprocessor,
        crop_size=crop_size,
        is_training=False,
        standardizer=standardizer,
        fill_mode='constant'
    )

    return prediction_iterator
