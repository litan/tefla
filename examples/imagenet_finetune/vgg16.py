from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout, relu
from tefla.core.layers import input, conv2d, fully_connected, max_pool, softmax

# sizes - (width, height)
image_size = (224, 224)
crop_size = (224, 224)


def model(is_training, reuse):
    common_args = common_layer_args(is_training, reuse)
    common_frozen_args = make_args(trainable=False, **common_args)
    conv_args = make_args(activation=relu, **common_args)
    conv_frozen_args = make_args(activation=relu, **common_frozen_args)
    pool_args = make_args(filter_size=(2, 2), **common_args)
    logit_args = make_args(activation=None, **common_args)

    x = input((None, crop_size[1], crop_size[0], 3), **common_args)

    with tf.variable_scope('vgg_16', reuse=reuse):
        mean_rgb = tf.get_variable(name='mean_rgb', initializer=tf.truncated_normal(shape=[3]), trainable=False)
        x = x - mean_rgb
        with tf.variable_scope('conv1'):
            x = conv2d(x, 64, name='conv1_1', **conv_frozen_args)
            x = conv2d(x, 64, name='conv1_2', **conv_frozen_args)
            x = max_pool(x, name='maxpool1', **pool_args)

        with tf.variable_scope('conv2'):
            x = conv2d(x, 128, name='conv2_1', **conv_frozen_args)
            x = conv2d(x, 128, name='conv2_2', **conv_frozen_args)
            x = max_pool(x, name='maxpool2', **pool_args)

        with tf.variable_scope('conv3'):
            x = conv2d(x, 256, name='conv3_1', **conv_frozen_args)
            x = conv2d(x, 256, name='conv3_2', **conv_frozen_args)
            x = conv2d(x, 256, name='conv3_3', **conv_frozen_args)
            x = max_pool(x, name='maxpool3', **pool_args)

        with tf.variable_scope('conv4'):
            x = conv2d(x, 512, name='conv4_1', **conv_frozen_args)
            x = conv2d(x, 512, name='conv4_2', **conv_frozen_args)
            x = conv2d(x, 512, name='conv4_3', **conv_frozen_args)
            x = max_pool(x, name='maxpool4', **pool_args)

        with tf.variable_scope('conv5'):
            x = conv2d(x, 512, name='conv5_1', **conv_frozen_args)
            x = conv2d(x, 512, name='conv5_2', **conv_frozen_args)
            x = conv2d(x, 512, name='conv5_3', **conv_frozen_args)
            x = max_pool(x, name='maxpool5', **pool_args)

        x = conv2d(x, 4096, name='fc6', filter_size=(7, 7), padding='VALID', **conv_frozen_args)
        x = dropout(x, drop_p=0.5, name='dropout6', **common_args)

        x = conv2d(x, 4096, name='fc7', filter_size=(1, 1), **conv_frozen_args)
        x = dropout(x, drop_p=0.5, name='dropout7', **common_args)

        logits = fully_connected(x, 2, name='logits', **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    return end_points(is_training)
