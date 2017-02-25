from __future__ import division, print_function, absolute_import

import logging
import os
import pprint
import time
import threading

import numpy as np
import tensorflow as tf

from tefla.core.lr_policy import NoDecayPolicy
from tefla.da.iterator import BatchIterator
from tefla.utils import util

logger = logging.getLogger('tefla')


class SupervisedTrainerN2(object):
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
        self._setup_misc()
        self._print_info(data_set, verbose)
        self._train_loop(data_set, weights_from, start_epoch, resume_lr, summary_every,
                         verbose, clean)

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)

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

                # util.show_layer_shapes(self.training_end_points, logger)

    def _get_from_queue(self, num_batchs_in_epoch):
        self.queue = tf.FIFOQueue(num_batchs_in_epoch, dtypes=[tf.float32, tf.int64],
                                  shapes=[[self.training_iterator.batch_size, 28, 28, 1],
                                          [self.training_iterator.batch_size, ]])
        self.batch_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.batch_y = tf.placeholder(tf.int64, shape=[None, ])
        self.enqueue_op = self.queue.enqueue([self.batch_x, self.batch_y])
        return self.queue.dequeue()

    def enqueue_thread_fn(self, sess, coord, batches):
        for batch_data in batches:
            if coord.should_stop():
                return
            if batch_data[0].shape[0] == self.training_iterator.batch_size:
                sess.run(self.enqueue_op, feed_dict={self.batch_x: batch_data[0], self.batch_y: batch_data[1]})

    def _train_loop(self, data_set, weights_from, start_epoch, resume_lr, summary_every,
                    verbose, clean):
        training_X, training_y, validation_X, validation_y = \
            data_set.training_X, data_set.training_y, data_set.validation_X, data_set.validation_y
        batch_iters_per_epoch = int(round(len(training_X) / self.training_iterator.batch_size))
        validation_batch_iters_per_epoch = int(round(len(validation_X) / self.validation_iterator.batch_size))
        weights_dir = "weights"
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        num_gpus = 2
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            optimizer = self._create_optimizer()
            tower_losses = []
            tower_grads = []
            tower_validation_predictions = []
            tower_validation_losses = []
            inputs, target = self._get_from_queue(num_batchs_in_epoch=batch_iters_per_epoch)
            gpu_inputs = tf.split(inputs, num_gpus, axis=0)
            gpu_targets = tf.split(target, num_gpus, axis=0)
            with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                for i in xrange(num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            loss, validation_predictions, validation_loss = self._tower_loss(gpu_inputs[i],
                                                                                             gpu_targets[i], scope)
                            var_scope.reuse_variables()
                            grads = optimizer.compute_gradients(loss)
                            tower_losses.append(loss)
                            tower_grads.append(grads)
                            tower_validation_predictions.append(validation_predictions)
                            tower_validation_losses.append(validation_loss)

            grads = _average_gradients(tower_grads)
            loss = tf.reduce_mean(tower_losses)
            apply_gradient_op = optimizer.apply_gradients(grads)
            train_op = apply_gradient_op

            validation_predictions = tf.concat(tower_validation_predictions, axis=0)
            validation_loss = tf.reduce_mean(tower_validation_losses)

            saver = tf.train.Saver(max_to_keep=None)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.cnf.get('gpu_memory_fraction', 0.9))
            with tf.Session(
                    config=tf.ConfigProto(
                        gpu_options=gpu_options,
                        allow_soft_placement=True,
                        log_device_placement=True
                    )) as sess:

                if start_epoch > 1:
                    weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()

                learning_rate_value = self.lr_policy.initial_lr
                if weights_from:
                    util.load_variables(sess, saver, weights_from, logger)
                    learning_rate_value = self.lr_policy.resume_lr(start_epoch, batch_iters_per_epoch, resume_lr)

                logger.info("Initial learning rate: %f " % learning_rate_value)

                seed_delta = 100
                training_history = []
                for epoch in xrange(start_epoch, self.num_epochs + 1):
                    np.random.seed(epoch + seed_delta)
                    tf.set_random_seed(epoch + seed_delta)
                    tic = time.time()
                    training_losses = []
                    batch_train_sizes = []

                    t = threading.Thread(target=self.enqueue_thread_fn,
                                         args=(sess, coord, self.training_iterator(training_X, training_y)))
                    t.start()
                    for batch_num in range(1, batch_iters_per_epoch):
                        feed_dict_train = {self.learning_rate: learning_rate_value}
                        training_loss_e, _ = sess.run([loss, train_op],
                                                      feed_dict=feed_dict_train)
                        training_losses.append(training_loss_e)
                        # batch_train_sizes.append(len(Xb))
                        batch_train_sizes.append(self.training_iterator.batch_size)

                        batch_iter_idx = (epoch - 1) * batch_iters_per_epoch + batch_num
                        learning_rate_value = self.lr_policy.batch_update(learning_rate_value, batch_iter_idx,
                                                                          batch_iters_per_epoch)

                    epoch_training_loss = np.average(training_losses, weights=batch_train_sizes)

                    # Validation prediction and metrics
                    # validation_losses = []
                    # batch_validation_metrics = [[] for _, _ in self.validation_metrics_def]
                    # epoch_validation_metrics = []
                    # batch_validation_sizes = []
                    #
                    # t = threading.Thread(target=self.enqueue_thread_fn,
                    #                      args=(sess, coord, self.validation_iterator(validation_X, validation_y)))
                    # t.start()
                    # for batch_num in range(1, validation_batch_iters_per_epoch):
                    #     validation_predictions_e, validation_loss_e = sess.run(
                    #         [validation_predictions, validation_loss],
                    #         feed_dict=None)
                    #     validation_losses.append(validation_loss_e)
                    #     # batch_validation_sizes.append(len(validation_Xb))
                    #     batch_validation_sizes.append(self.validation_iterator.batch_size)
                    #     # for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                    #     #     metric_score = metric_function(validation_yb, validation_predictions_e)
                    #     #     batch_validation_metrics[i].append(metric_score)
                    #
                    # epoch_validation_loss = np.average(validation_losses, weights=batch_validation_sizes)
                    # # for i, (_, _) in enumerate(self.validation_metrics_def):
                    # #     epoch_validation_metrics.append(
                    # #         np.average(batch_validation_metrics[i], weights=batch_validation_sizes))
                    # #
                    # # custom_metrics_string = [', %s: %.4f' % (name, epoch_validation_metrics[i]) for i, (name, _) in
                    # #                          enumerate(self.validation_metrics_def)]
                    # # custom_metrics_string = ''.join(custom_metrics_string)

                    logger.info(
                        "Epoch %d [(%s, %s) images, %6.1fs]: t-loss: %.3f" %
                        (epoch, np.sum(batch_train_sizes), -1, time.time() - tic,
                         epoch_training_loss,
                         )
                    )

                    saver.save(sess, "%s/model-epoch-%d.ckpt" % (weights_dir, epoch))

                    epoch_info = dict(
                        epoch=epoch,
                        training_loss=epoch_training_loss,
                        validation_loss=-1
                    )

                    training_history.append(epoch_info)
                    learning_rate_value = self.lr_policy.epoch_update(learning_rate_value, training_history)

    def _create_optimizer(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")
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

    def _tower_loss(self, inputs, target, scope):
        training_end_points = self.model(is_training=True, reuse=None, inputs=inputs)
        training_logits, training_predictions = training_end_points['logits'], training_end_points['predictions']
        validation_end_points = self.model(is_training=False, reuse=True, inputs=inputs)
        validation_logits, validation_predictions = validation_end_points['logits'], \
                                                    validation_end_points['predictions']

        with tf.name_scope('loss'):
            training_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=training_logits, labels=target))

            validation_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=validation_logits, labels=target))

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), scope)
            return training_loss + l2_loss * self.cnf.get('l2_reg', 0.0), validation_predictions, validation_loss

    def _adjust_ground_truth(self, y):
        return y if self.classification else y.reshape(-1, 1).astype(np.float32)


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
