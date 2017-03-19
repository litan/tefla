from __future__ import division, print_function, absolute_import

import click
import numpy as np

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tefla.core.dir_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.core.training_q import SupervisedTrainerQ
from tefla.utils import util
import logging
import sys


@click.command()
@click.option('--model', help='Relative path to model.')
@click.option('--training_cnf', help='Relative path to training config file.')
@click.option('--data_dir', help='Path to training directory.')
@click.option('--start_epoch', default=1, show_default=True,
              help='Epoch number from which to resume training.')
@click.option('--resume_lr', help='Learning rate for resumed training.')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--weights_exclude_scopes', help='Scopes not to load from weights file.')
@click.option('--trainable_scopes', help='Scopes to train.')
@click.option('--clean', is_flag=True, help='Clean out training log and summary dir.')
@click.option('--visuals', is_flag=True, help='Visualize your training using various graphs.')
def main(model, training_cnf, data_dir, start_epoch, resume_lr, weights_from, weights_exclude_scopes, trainable_scopes,
         clean, visuals):
    util.check_required_program_args([model, training_cnf, data_dir])
    sys.path.insert(0, '.')
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf

    util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO, clean=clean)
    if weights_from:
        weights_from = str(weights_from)

    data_set = DataSet(data_dir, model_def.image_size[0])
    training_iter, validation_iter = create_training_iters(cnf, data_set, model_def.crop_size, start_epoch,
                                                           cnf.get('iterator_type', 'parallel') == 'parallel')

    try:
        input_shape = (-1, model_def.crop_size[1], model_def.crop_size[0], model_def.num_channels)
    except AttributeError:
        input_shape = (-1, model_def.crop_size[1], model_def.crop_size[0], 3)

    trainer = SupervisedTrainerQ(model, cnf, input_shape, trainable_scopes, training_iter, validation_iter,
                                 classification=cnf['classification'])
    trainer.fit(data_set, weights_from, weights_exclude_scopes, start_epoch, resume_lr, verbose=1,
                summary_every=cnf.get('summary_every', 10), clean=clean, visuals=visuals)


if __name__ == '__main__':
    main()
