from __future__ import division, print_function, absolute_import

import click
import numpy as np

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tefla.core.dir_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.core.training import SupervisedTrainer
from tefla.utils import util
import logging


@click.command()
@click.option('--model', help='Relative path to model.')
@click.option('--training_cnf', help='Relative path to training config file.')
@click.option('--data_dir', help='Path to training directory.')
@click.option('--start_epoch', default=1, show_default=True,
              help='Epoch number from which to resume training.')
@click.option('--resume_lr', help='Learning rate for resumed training.')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--clean', is_flag=True, help='Clean out training log and summary dir.')
def main(model, training_cnf, data_dir, start_epoch, resume_lr, weights_from, clean):
    util.check_required_program_args([model, training_cnf, data_dir])
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf

    util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO, clean=clean)
    if weights_from:
        weights_from = str(weights_from)

    data_set = DataSet(data_dir, model_def.image_size[0])
    training_iter, validation_iter = create_training_iters(cnf, data_set, model_def.crop_size, start_epoch,
                                                           cnf.get('iterator_type', 'queued') == 'parallel')
    trainer = SupervisedTrainer(model, cnf, training_iter, validation_iter, classification=cnf['classification'])
    trainer.fit(data_set, weights_from, start_epoch, resume_lr, verbose=1,
                summary_every=cnf.get('summary_every', 10), clean=clean)


if __name__ == '__main__':
    main()
