# Ported from https://github.com/tensorflow/tensorflow/tree/r0.12/tensorflow/contrib/slim/python/slim/nets
from __future__ import division, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils

from tefla.core.layer_arg_ops import make_args, end_points, common_layer_args
from tefla.core.layers import conv2d, conv2d_same, max_pool, softmax, batch_norm_tf, input, squeeze
from tefla.core.layers import relu, collect_named_outputs, alias_scoped

# sizes - (width, height)
image_size = (224, 224)
crop_size = (224, 224)

batch_norm_params = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
}


def subsample(inputs, factor, scope=None, **common_args):
    if factor == 1:
        return alias_scoped(inputs, name=scope, **common_args)
    else:
        return max_pool(inputs, filter_size=(1, 1), stride=(factor, factor), padding='SAME', name=scope, **common_args)


def make_conv_args(activation, **common_args):
    return make_args(use_bias=False, batch_norm=batch_norm_tf, batch_norm_args=batch_norm_params,
                     activation=activation, **common_args)


def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               scope=None, **common_args):
    conv_args_none = make_conv_args(activation=None, **common_args)
    conv_args_relu = make_conv_args(activation=relu, **common_args)
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut', **common_args)
        else:
            shortcut = conv2d(inputs, depth, filter_size=(1, 1), stride=(stride, stride),
                              name='shortcut', **conv_args_none)

        residual = conv2d(inputs, depth_bottleneck, filter_size=(1, 1), stride=(1, 1),
                          name='conv1', **conv_args_relu)
        residual = conv2d_same(residual, depth_bottleneck, filter_size=(3, 3), stride=(stride, stride),
                               dilation_rate=rate, name='conv2', **conv_args_relu)
        residual = conv2d(residual, depth, filter_size=(1, 1), stride=(1, 1),
                          name='conv3', **conv_args_none)

        # not in endpoints. does that matter?
        output = tf.nn.relu(shortcut + residual)

        return collect_named_outputs(common_args['outputs_collections'],
                                     sc.name,
                                     output)


def model(is_training, reuse, inputs=None):
    common_trainable_args = common_layer_args(is_training, reuse, trainable=True)
    common_frozen_args = common_layer_args(is_training, reuse, trainable=False)
    conv_args = make_conv_args(activation=relu, **common_frozen_args)
    logit_args = make_args(activation=None, **common_trainable_args)

    common_args = common_frozen_args
    # move this down to train only a few layers
    common_args = common_trainable_args
    if inputs is None:
        net = input((None, crop_size[1], crop_size[0], 3), **common_args)
    else:
        net = inputs
    with tf.variable_scope('resnet_v1_50', reuse=reuse):
        mean_rgb = tf.get_variable(name='mean_rgb', initializer=tf.truncated_normal(shape=[3]), trainable=False)
        net = net - mean_rgb
        net = conv2d_same(net, 64, filter_size=(7, 7), stride=(2, 2), name='conv1', **conv_args)
        net = max_pool(net, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool1')

        with tf.variable_scope('block1') as sc:
            with tf.variable_scope('unit_1'):
                net = bottleneck(net, 256, 64, 1, **common_args)
            with tf.variable_scope('unit_2'):
                net = bottleneck(net, 256, 64, 1, **common_args)
            with tf.variable_scope('unit_3'):
                net = bottleneck(net, 256, 64, 2, **common_args)
            net = collect_named_outputs(common_args['outputs_collections'], sc.name, net)

        with tf.variable_scope('block2') as sc:
            with tf.variable_scope('unit_1'):
                net = bottleneck(net, 512, 128, 1, **common_args)
            with tf.variable_scope('unit_2'):
                net = bottleneck(net, 512, 128, 1, **common_args)
            with tf.variable_scope('unit_3'):
                net = bottleneck(net, 512, 128, 1, **common_args)
            with tf.variable_scope('unit_4'):
                net = bottleneck(net, 512, 128, 2, **common_args)
            net = collect_named_outputs(common_args['outputs_collections'], sc.name, net)

        with tf.variable_scope('block3') as sc:
            with tf.variable_scope('unit_1'):
                net = bottleneck(net, 1024, 256, 1, **common_args)
            with tf.variable_scope('unit_2'):
                net = bottleneck(net, 1024, 256, 1, **common_args)
            with tf.variable_scope('unit_3'):
                net = bottleneck(net, 1024, 256, 1, **common_args)
            with tf.variable_scope('unit_4'):
                net = bottleneck(net, 1024, 256, 1, **common_args)
            with tf.variable_scope('unit_5'):
                net = bottleneck(net, 1024, 256, 1, **common_args)
            with tf.variable_scope('unit_6'):
                net = bottleneck(net, 1024, 256, 2, **common_args)
            net = collect_named_outputs(common_args['outputs_collections'], sc.name, net)

        with tf.variable_scope('block4') as sc:
            with tf.variable_scope('unit_1'):
                net = bottleneck(net, 2048, 512, 1, **common_args)
            with tf.variable_scope('unit_2'):
                net = bottleneck(net, 2048, 512, 1, **common_args)
            with tf.variable_scope('unit_3'):
                net = bottleneck(net, 2048, 512, 1, **common_args)
            net = collect_named_outputs(common_args['outputs_collections'], sc.name, net)

        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        net = conv2d(net, 1000, filter_size=(1, 1), name='logits', **logit_args)
        logits = squeeze(net, axis=[1, 2], name='logits', **common_args)

    predictions = softmax(logits, name='predictions', **common_args)
    return end_points(common_args['is_training'])
