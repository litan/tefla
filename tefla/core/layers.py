from __future__ import division, print_function, absolute_import

from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

from tefla.core import initializers as initz
from tefla.utils import util as helper

NamedOutputs = namedtuple('NamedOutputs', ['name', 'outputs'])


def input(shape, outputs_collections=None, name='inputs', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        inputs = tf.placeholder(tf.float32, shape=shape, name="input")
        return _collect_named_outputs(outputs_collections, curr_scope, inputs)


def alias(x, outputs_collections=None, name='alias', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        return _collect_named_outputs(outputs_collections, curr_scope, x)


def reshape(x, shape, outputs_collections=None, name='reshape', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        x = tf.reshape(x, shape)
        return _collect_named_outputs(outputs_collections, curr_scope, x)


def squeeze(x, axis, outputs_collections=None, name='squeeze', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        x = tf.squeeze(x, axis=axis)
        return _collect_named_outputs(outputs_collections, curr_scope, x)


def fully_connected(x, n_output, is_training, reuse, activation=None, batch_norm=None, batch_norm_args=None,
                    w_init=initz.he_normal(), use_bias=True, b_init=0.0, w_regularizer=tf.nn.l2_loss,
                    outputs_collections=None, trainable=True, name='fc'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    if len(x.get_shape()) != 2:
        x = _flatten(x)

    n_input = helper.get_input_shape(x)[1]

    with tf.variable_scope(name, reuse=reuse) as curr_scope:
        shape = [n_input, n_output] if hasattr(w_init, '__call__') else None
        W = tf.get_variable(
            name='weights',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )
        output = tf.matmul(x, W)

        if use_bias:
            b = tf.get_variable(
                name='biases',
                shape=[n_output],
                initializer=tf.constant_initializer(b_init),
                trainable=trainable
            )

            output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training, reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, is_training=is_training, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, curr_scope.original_name_scope, output)


def conv2d(x, n_output_channels, is_training, reuse, filter_size=(3, 3), stride=(1, 1), dilation_rate=1,
           padding='SAME', activation=None, batch_norm=None, batch_norm_args=None, w_init=initz.he_normal(),
           use_bias=True, untie_biases=False, b_init=0.0, w_regularizer=tf.nn.l2_loss,
           outputs_collections=None, trainable=True, name='conv2d'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse) as curr_scope:
        shape = [filter_size[0], filter_size[1], x.get_shape()[-1], n_output_channels] if hasattr(w_init,
                                                                                                  '__call__') else None
        W = tf.get_variable(
            name='weights',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        if dilation_rate == 1:
            output = tf.nn.conv2d(
                input=x,
                filter=W,
                strides=[1, stride[0], stride[1], 1],
                padding=padding)
        else:
            if len([s for s in stride if s > 1]) > 0:
                raise ValueError("Stride (%s) cannot be more than 1 if rate (%d) is not 1" % (stride, dilation_rate))

            output = tf.nn.atrous_conv2d(
                value=x,
                filters=W,
                rate=dilation_rate,
                padding=padding)

        if use_bias:
            if untie_biases:
                b = tf.get_variable(
                    name='biases',
                    shape=output.get_shape()[1:],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable
                )
                output = tf.add(output, b)
            else:
                b = tf.get_variable(
                    name='biases',
                    shape=[n_output_channels],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable
                )
                output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training, reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, is_training=is_training, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, curr_scope.original_name_scope, output)


def max_pool(x, filter_size=(3, 3), stride=(2, 2), padding='VALID', outputs_collections=None, name='max_pool',
             **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name) as curr_scope:
        output = tf.nn.max_pool(
            value=x,
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
        )
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def rms_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', epsilon=0.000000000001,
                outputs_collections=None, name='rms_pool', **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name) as curr_scope:
        output = tf.nn.avg_pool(
            value=tf.square(x),
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
        )
        output = tf.sqrt(output + epsilon)
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def avg_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', outputs_collections=None, name='avg_pool',
                **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name) as curr_scope:
        output = tf.nn.avg_pool(
            value=x,
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
        )
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def global_avg_pool(x, outputs_collections=None, name="global_avg_pool", **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name) as curr_scope:
        output = tf.reduce_mean(x, [1, 2])
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def feature_max_pool_1d(x, stride=2, outputs_collections=None, name='feature_max_pool', **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 2, "Input Tensor shape must be 2-D"
    with tf.name_scope(name) as curr_scope:
        x = tf.reshape(x, (-1, input_shape[1] // stride, stride))
        output = tf.reduce_max(
            input_tensor=x,
            reduction_indices=[2],
        )
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def batch_norm_tf(x, scale=False, updates_collections=None, name='BatchNorm', **kwargs):
    outputs_collection = kwargs.pop('outputs_collections', None)
    output = tf.contrib.layers.batch_norm(x, scope=name, scale=scale, outputs_collections=None,
                                          updates_collections=updates_collections, **kwargs)
    return _collect_named_outputs(outputs_collection, output.aliases[0], output)


def batch_norm_lasagne(x, is_training, reuse, decay=0.9, epsilon=1e-4, updates_collections=tf.GraphKeys.UPDATE_OPS,
                       outputs_collections=None, trainable=True, name='bn'):
    with tf.variable_scope(name, reuse=reuse) as curr_scope:
        beta = tf.get_variable(
            name='beta',
            initializer=tf.constant(0.0, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )
        gamma = tf.get_variable(
            name='gamma',
            initializer=tf.constant(1.0, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )

        moving_mean = tf.get_variable(
            name='moving_mean',
            shape=[x.get_shape()[-1]],
            initializer=tf.zeros_initializer,
            trainable=False)

        moving_inv_std = tf.get_variable(
            name='moving_inv_std',
            shape=[x.get_shape()[-1]],
            initializer=tf.ones_initializer(),
            trainable=False)

        input_shape = helper.get_input_shape(x)
        moments_axes = list(range(len(input_shape) - 1))

        def mean_inv_std_with_update():
            mean, variance = tf.nn.moments(x, moments_axes, shift=moving_mean, name='bn-moments')
            inv_std = math_ops.rsqrt(variance + epsilon)
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False)
            update_moving_inv_std = moving_averages.assign_moving_average(
                moving_inv_std, inv_std, decay, zero_debias=False)
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_inv_std]):
                m, v = tf.identity(mean), tf.identity(inv_std)
                return m, v

        def mean_inv_std_with_pending_update():
            mean, variance = tf.nn.moments(x, moments_axes, shift=moving_mean, name='bn-moments')
            inv_std = math_ops.rsqrt(variance + epsilon)
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False)
            update_moving_inv_std = moving_averages.assign_moving_average(
                moving_inv_std, inv_std, decay, zero_debias=False)
            tf.add_to_collection(updates_collections, update_moving_mean)
            tf.add_to_collection(updates_collections, update_moving_inv_std)
            return mean, inv_std

        mean_inv_std_with_relevant_update = \
            mean_inv_std_with_pending_update if updates_collections is not None else mean_inv_std_with_update

        (mean, inv_std) = mean_inv_std_with_relevant_update() if is_training else (moving_mean, moving_inv_std)

        def _batch_normalization(x, mean, inv, offset, scale):
            with tf.name_scope(name, "batchnorm", [x, mean, inv, scale, offset]):
                if scale is not None:
                    inv *= scale
                return x * inv + (offset - mean * inv
                                  if offset is not None else -mean * inv)

        output = _batch_normalization(x, mean, inv_std, beta, gamma)
        return _collect_named_outputs(outputs_collections, curr_scope.original_name_scope, output)


def prelu(x, reuse, outputs_collections=None, trainable=True, name='prelu', **unused):
    _check_unused(unused, name)
    with tf.variable_scope(name, reuse=reuse) as curr_scope:
        alphas = tf.get_variable(
            name='alpha',
            initializer=tf.constant(0.2, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )

        output = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
        return _collect_named_outputs(outputs_collections, curr_scope.original_name_scope, output)


def relu(x, outputs_collections=None, name='relu', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        output = tf.nn.relu(x)
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def leaky_relu(x, alpha=0.01, outputs_collections=None, name='leaky_relu', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        output = tf.nn.relu(x) + tf.multiply(alpha, (x - tf.abs(x))) * 0.5
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def softmax(x, outputs_collections=None, name='softmax', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        output = tf.nn.softmax(x)
        return _collect_named_outputs(outputs_collections, curr_scope, output)


def dropout(x, is_training, drop_p=0.5, outputs_collections=None, name='dropout', **unused):
    _check_unused(unused, name)
    with tf.name_scope(name) as curr_scope:
        keep_p = 1. - drop_p
        if is_training:
            output = tf.nn.dropout(x, keep_p, seed=None)
            return _collect_named_outputs(outputs_collections, curr_scope, output)
        else:
            return _collect_named_outputs(outputs_collections, curr_scope, x)


def _flatten(x):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    dims = int(np.prod(input_shape[1:]))
    flattened = tf.reshape(x, [-1, dims])
    return flattened


def _collect_named_outputs(outputs_collections, name, output):
    if name[-1] == '/':
        name = name[:-1]
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, NamedOutputs(name, output))
    return output


def _check_unused(unused, name):
    allowed_keys = ['is_training', 'reuse', 'outputs_collections', 'trainable']
    helper.veryify_args(unused, allowed_keys, 'Layer "%s" got unexpected argument(s):' % name)
