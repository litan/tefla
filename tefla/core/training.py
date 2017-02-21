from __future__ import division, print_function, absolute_import

import logging
import os
import pprint
import shutil
import time

import numpy as np
import tensorflow as tf
from tefla.utils import store_training_logs
from tefla.core.lr_policy import NoDecayPolicy
from tefla.da.iterator import BatchIterator
from tefla.utils import util

logger = logging.getLogger('tefla')

TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'

class SupervisedTrainer(object):
    def __init__(self, model, cnf, training_iterator=BatchIterator(32, False),
                 validation_iterator=BatchIterator(128, False), classification=True, clip_norm=False):
        self.model = model
        self.cnf = cnf
        self.training_iterator = training_iterator
        self.validation_iterator = validation_iterator
        self.classification = classification
        self.lr_policy = cnf.get('lr_policy', NoDecayPolicy(0.01))
        self.validation_metrics_def = self.cnf.get('validation_scores', [])
        self.clip_norm = clip_norm

    def fit(self, data_set, weights_from=None, start_epoch=1, resume_lr=None, summary_every=10, verbose=0, clean=False):
        self._setup_predictions_and_loss()
        self._setup_optimizer()
        self._setup_summaries()
        self._setup_misc()
        self._print_info(data_set, verbose)
        self._train_loop(data_set, weights_from, start_epoch, resume_lr, summary_every,
                         verbose, clean)

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None and len(self.update_ops) == 0:
            self.update_ops = None
            # if update_ops is not None:
            #     regularized_training_loss = control_flow_ops.with_dependencies(update_ops, regularized_training_loss)

    def _print_info(self, data_set, verbose):
        logger.info('Config:')
        logger.info(pprint.pformat(self.cnf))
        data_set.print_info()
        logger.info('Max epochs: %d' % self.num_epochs)
        if verbose > 0:
            util.show_vars(logger)

        # logger.debug("\n---Number of Regularizable vars in model:")
        # logger.debug(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        if verbose > 3:
            all_ops = tf.get_default_graph().get_operations()
            logger.debug("\n---All ops in graph")
            names = map(lambda v: v.name, all_ops)
            for n in sorted(names):
                logger.debug(n)

        util.show_layer_shapes(self.training_end_points, logger)

    def _train_loop(self, data_set, weights_from, start_epoch, resume_lr, summary_every,
                    verbose, clean):
        training_X, training_y, validation_X, validation_y = \
            data_set.training_X, data_set.training_y, data_set.validation_X, data_set.validation_y
        saver = tf.train.Saver(max_to_keep=None)
        weights_dir = "weights"
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        training_batch_summary_op = tf.summary.merge_all(key=TRAINING_BATCH_SUMMARIES)
        training_epoch_summary_op = tf.summary.merge_all(key=TRAINING_EPOCH_SUMMARIES)
        validation_batch_summary_op = tf.summary.merge_all(key=VALIDATION_BATCH_SUMMARIES)
        validation_epoch_summary_op = tf.summary.merge_all(key=VALIDATION_EPOCH_SUMMARIES)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.cnf.get('gpu_memory_fraction', 0.9))
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

            sess.run(tf.global_variables_initializer())
            batch_iters_per_epoch = int(round(len(data_set.training_X) / self.training_iterator.batch_size))
            learning_rate_value = self.lr_policy.initial_lr
            if weights_from:
                util.load_variables(sess, saver, weights_from, logger)
                learning_rate_value = self.lr_policy.resume_lr(start_epoch, batch_iters_per_epoch, resume_lr)

            logger.info("Initial learning rate: %f " % learning_rate_value)
            train_writer, validation_writer = _create_summary_writer(self.cnf.get('summary_dir'),
                                                                     sess, clean)

            seed_delta = 100
            training_history = []
            input = store_training_logs.delete_file('run_script_logs.pkl')
            for epoch in xrange(start_epoch, self.num_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                training_losses = []
                batch_train_sizes = []

                for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y), start=1):
                    feed_dict_train = {self.inputs: Xb, self.target: self._adjust_ground_truth(yb),
                                       self.learning_rate: learning_rate_value}

                    logger.debug('1. Loading batch %d data done.' % batch_num)
                    if (epoch - 1) % summary_every == 0 and batch_num < 2:
                        logger.debug('2. Running training steps with summary...')
                        training_predictions_e, training_loss_e, summary_str_train, _ = sess.run(
                            [self.training_predictions, self.regularized_training_loss, training_batch_summary_op,
                             self.optimizer_step],
                            feed_dict=feed_dict_train)
                        train_writer.add_summary(summary_str_train, epoch)
                        train_writer.flush()
                        logger.debug('2. Running training steps with summary done.')
                        if verbose > 3:
                            logger.debug("Epoch %d, Batch %d training loss: %s" % (epoch, batch_num, training_loss_e))
                            logger.debug("Epoch %d, Batch %d training predictions: %s" %
                                         (epoch, batch_num, training_predictions_e))
                    else:
                        logger.debug('2. Running training steps without summary...')
                        training_loss_e, _ = sess.run([self.regularized_training_loss, self.optimizer_step],
                                                      feed_dict=feed_dict_train)
                        logger.debug('2. Running training steps without summary done.')

                    training_losses.append(training_loss_e)
                    batch_train_sizes.append(len(Xb))

                    if self.update_ops is not None:
                        logger.debug('3. Running update ops...')
                        sess.run(self.update_ops, feed_dict=feed_dict_train)
                        logger.debug('3. Running update ops done.')

                    batch_iter_idx = (epoch - 1) * batch_iters_per_epoch + batch_num
                    learning_rate_value = self.lr_policy.batch_update(learning_rate_value, batch_iter_idx,
                                                                      batch_iters_per_epoch)
                    logger.debug('4. Training batch %d done.' % batch_num)

                epoch_training_loss = np.average(training_losses, weights=batch_train_sizes)

                # Plot training loss every epoch
                logger.debug('5. Writing epoch summary...')
                summary_str_train = sess.run(training_epoch_summary_op,
                                             feed_dict={self.epoch_loss: epoch_training_loss,
                                                        self.learning_rate: learning_rate_value})
                train_writer.add_summary(summary_str_train, epoch)
                train_writer.flush()
                logger.debug('5. Writing epoch summary done.')

                # Validation prediction and metrics
                validation_losses = []
                batch_validation_metrics = [[] for _, _ in self.validation_metrics_def]
                epoch_validation_metrics = []
                batch_validation_sizes = []
                for batch_num, (validation_Xb, validation_yb) in enumerate(
                        self.validation_iterator(validation_X, validation_y), start=1):
                    feed_dict_validation = {self.validation_inputs: validation_Xb,
                                            self.target: self._adjust_ground_truth(validation_yb)}
                    logger.debug('6. Loading batch %d validation data done.' % batch_num)

                    if (epoch - 1) % summary_every == 0 and batch_num < 2:
                        logger.debug('7. Running validation steps with summary...')
                        validation_predictions_e, validation_loss_e, summary_str_validate = sess.run(
                            [self.validation_predictions, self.validation_loss, validation_batch_summary_op],
                            feed_dict=feed_dict_validation)
                        validation_writer.add_summary(summary_str_validate, epoch)
                        validation_writer.flush()
                        logger.debug('7. Running validation steps with summary done.')
                        if verbose > 3:
                            logger.debug(
                                "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, validation_loss_e))
                            logger.debug("Epoch %d, Batch %d validation predictions: %s" % (
                                epoch, batch_num, validation_predictions_e))
                    else:
                        logger.debug('7. Running validation steps without summary...')
                        validation_predictions_e, validation_loss_e = sess.run(
                            [self.validation_predictions, self.validation_loss],
                            feed_dict=feed_dict_validation)
                        logger.debug('7. Running validation steps without summary done.')
                    validation_losses.append(validation_loss_e)
                    batch_validation_sizes.append(len(validation_Xb))

                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(validation_yb, validation_predictions_e)
                        batch_validation_metrics[i].append(metric_score)
                    logger.debug('8. Validation batch %d done' % batch_num)

                epoch_validation_loss = np.average(validation_losses, weights=batch_validation_sizes)
                for i, (_, _) in enumerate(self.validation_metrics_def):
                    epoch_validation_metrics.append(
                        np.average(batch_validation_metrics[i], weights=batch_validation_sizes))

                # Write validation epoch summary every epoch
                logger.debug('9. Writing epoch validation summary...')
                summary_str_validate = sess.run(validation_epoch_summary_op,
                                                feed_dict={self.epoch_loss: epoch_validation_loss,
                                                           self.validation_metric_placeholders: epoch_validation_metrics})
                validation_writer.add_summary(summary_str_validate, epoch)
                validation_writer.flush()
                logger.debug('9. Writing epoch validation summary done.')

                custom_metrics_string = [', %s: %.4f' % (name, epoch_validation_metrics[i]) for i, (name, _) in
                                         enumerate(self.validation_metrics_def)]
                custom_metrics_string = ''.join(custom_metrics_string)

                logger.info(
                    "Epoch %d [(%s, %s) images, %6.1fs]: t-loss: %.3f, v-loss: %.3f, t-loss/v-loss: %.3f%s" %
                    (epoch, np.sum(batch_train_sizes), np.sum(batch_validation_sizes), time.time() - tic,
                     epoch_training_loss,
                     epoch_validation_loss,
                     epoch_training_loss / epoch_validation_loss,
                     custom_metrics_string)
                )

                if input =='y':
                    store_training_logs.store_logs(epoch,epoch_validation_metrics[0],epoch_validation_metrics[1],epoch_training_loss,epoch_validation_loss,epoch_training_loss / epoch_validation_loss)

                saver.save(sess, "%s/model-epoch-%d.ckpt" % (weights_dir, epoch))

                epoch_info = dict(
                    epoch=epoch,
                    training_loss=epoch_training_loss,
                    validation_loss=epoch_validation_loss
                )

                training_history.append(epoch_info)

                learning_rate_value = self.lr_policy.epoch_update(learning_rate_value, training_history)
                if verbose > 0:
                    logger.info("Next epoch learning rate: %f " % learning_rate_value)
                logger.debug('10. Epoch done. [%d]' % epoch)

            train_writer.close()
            validation_writer.close()

    def _setup_summaries(self):
        def image_batch_like(shape):
            return len(shape) == 4 and shape[3] in {1, 3, 4}

        with tf.name_scope('summaries'):
            self.epoch_loss = tf.placeholder(tf.float32, shape=[], name="epoch_loss")

            # Training summaries
            tf.summary.scalar('learning rate', self.learning_rate, collections=[TRAINING_EPOCH_SUMMARIES])
            tf.summary.scalar('training (cross entropy) loss', self.epoch_loss,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            if image_batch_like(util.get_input_shape(self.inputs)):
                tf.summary.image('input', self.inputs, 10, collections=[TRAINING_BATCH_SUMMARIES])
            for key, val in self.training_end_points.iteritems():
                variable_summaries(val, key, collections=[TRAINING_BATCH_SUMMARIES])
            # for var in tf.trainable_variables():
            #     variable_summaries(var, var.op.name, collections=[TRAINING_BATCH_SUMMARIES])
            for grad, var in self.grads_and_vars:
                variable_summaries(var, var.op.name, collections=[TRAINING_BATCH_SUMMARIES])
                variable_summaries(grad, var.op.name + '/grad', collections=[TRAINING_BATCH_SUMMARIES])

            # Validation summaries
            for key, val in self.validation_end_points.iteritems():
                variable_summaries(val, key, collections=[VALIDATION_BATCH_SUMMARIES])

            tf.summary.scalar('validation loss', self.epoch_loss, collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = []
            for metric_name, _ in self.validation_metrics_def:
                validation_metric = tf.placeholder(tf.float32, shape=[], name=metric_name.replace(' ', '_'))
                self.validation_metric_placeholders.append(validation_metric)
                tf.summary.scalar(metric_name, validation_metric,
                                  collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = tuple(self.validation_metric_placeholders)

    def _setup_optimizer(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")
        # Keep old variable around to load old params, till we need this
        self.obsolete_learning_rate = tf.Variable(1.0, trainable=False, name="learning_rate")
        optimizer = self._create_optimizer()
        self.grads_and_vars = optimizer.compute_gradients(self.regularized_training_loss, tf.trainable_variables())
        if self.clip_norm:
            self.grads_and_vars = _clip_grad_norms(self.grads_and_vars)
        self.optimizer_step = optimizer.apply_gradients(self.grads_and_vars)

    def _create_optimizer(self):
        optimizer = self.cnf.get('optimizer')
        if optimizer is None:
            optimizer = tf.train.MomentumOptimizer(
                self.learning_rate,
                momentum=0.9,
                use_nesterov=True)
        else:
            if hasattr(optimizer, '_learning_rate'):
                optimizer._learning_rate = self.learning_rate
            elif hasattr(optimizer, '_lr'):
                optimizer._lr = self.learning_rate
            else:
                raise ValueError("Unknown optimizer")

        return optimizer

    def _setup_predictions_and_loss(self):
        if self.classification:
            self._setup_classification_predictions_and_loss()
        else:
            self._setup_regression_predictions_and_loss()

    def _setup_classification_predictions_and_loss(self):
        self.training_end_points = self.model(is_training=True, reuse=None)
        self.inputs = self.training_end_points['inputs']
        training_logits, self.training_predictions = self.training_end_points['logits'], self.training_end_points[
            'predictions']
        self.validation_end_points = self.model(is_training=False, reuse=True)
        # beware - we're depending on _1 suffixes based on name scopes here
        self.validation_inputs = self.validation_end_points['inputs_1']
        validation_logits, self.validation_predictions = self.validation_end_points['logits_1'], \
                                                         self.validation_end_points[
                                                             'predictions_1']
        with tf.name_scope('predictions'):
            self.target = tf.placeholder(tf.int32, shape=(None,), name='target')
        with tf.name_scope('loss'):
            training_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=training_logits, labels=self.target))

            self.validation_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=validation_logits, labels=self.target))

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularized_training_loss = training_loss + l2_loss * self.cnf.get('l2_reg', 0.0)

    def _setup_regression_predictions_and_loss(self):
        self.training_end_points = self.model(is_training=True, reuse=None)
        self.inputs = self.training_end_points['inputs']
        self.training_predictions = self.training_end_points['predictions']
        self.validation_end_points = self.model(is_training=False, reuse=True)
        self.validation_inputs = self.validation_end_points['inputs_1']
        self.validation_predictions = self.validation_end_points['predictions_1']
        with tf.name_scope('predictions'):
            self.target = tf.placeholder(tf.float32, shape=(None, 1), name='target')
        with tf.name_scope('loss'):
            training_loss = tf.reduce_mean(
                tf.square(tf.subtract(self.training_predictions, self.target)))

            self.validation_loss = tf.reduce_mean(
                tf.square(tf.subtract(self.validation_predictions, self.target)))

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularized_training_loss = training_loss + l2_loss * self.cnf.get('l2_reg', 0.0)

    def _adjust_ground_truth(self, y):
        return y if self.classification else y.reshape(-1, 1).astype(np.float32)


def _create_summary_writer(summary_dir, sess, clean):
    if summary_dir is None:
        summary_dir = '/tmp/tefla-summary'
        clean = True

    if clean and os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        os.mkdir(summary_dir + '/training')
        os.mkdir(summary_dir + '/validation')

    train_writer = tf.summary.FileWriter(summary_dir + '/training', graph=sess.graph)
    val_writer = tf.summary.FileWriter(summary_dir + '/validation', graph=sess.graph)
    return train_writer, val_writer


def variable_summaries(var, name, collections, extensive=False):
    if extensive:
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean, collections=collections)
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev, collections=collections)
        tf.summary.scalar('max/' + name, tf.reduce_max(var), collections=collections)
        tf.summary.scalar('min/' + name, tf.reduce_min(var), collections=collections)
    return tf.summary.histogram(name, var, collections=collections)


def _clip_grad_norms(self, gradients_to_variables, max_norm=5):
    """Clips the gradients by the given value.

    Args:
        gradients_to_variables: A list of gradient to variable pairs (tuples).
        max_norm: the maximum norm value.

    Returns:
        A list of clipped gradient to variable pairs.
     """
    grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        grads_and_vars.append((grad, var))
    return grads_and_vars
