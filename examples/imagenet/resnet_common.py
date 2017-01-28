# Ported from https://github.com/tensorflow/tensorflow/tree/r0.12/tensorflow/contrib/slim/python/slim/nets
from __future__ import division, print_function, absolute_import

import collections

import tensorflow as tf
from tensorflow.contrib.layers import utils

from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import input, conv2d, max_pool, softmax, batch_norm_tf
from tefla.core.layers import relu, _collect_named_outputs


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    pass


def subsample(inputs, factor, scope=None, **common_args):
    if factor == 1:
        return inputs
    else:
        return max_pool(inputs, filter_size=(1, 1), stride=(factor, factor), padding='SAME', name=scope)


batch_norm_params = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
}


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None, **common_args):
    assert rate == 1 # Todo - deal with dilated convolutions later
    # return conv2d(inputs, num_outputs, filter_size=(kernel_size, kernel_size), stride=(stride, stride),
    #               padding='SAME', name=scope, **common_args)
    if stride == 1:
        return conv2d(inputs, num_outputs, filter_size=(kernel_size, kernel_size), stride=(stride, stride),
                      padding='SAME', name=scope, **common_args)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return conv2d(inputs, num_outputs, filter_size=(kernel_size, kernel_size), stride=(stride, stride),
                      padding='VALID', name=scope, **common_args)


def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               scope=None, **common_args):
    conv_args_common = make_args(use_bias=False, batch_norm=batch_norm_tf, batch_norm_args=batch_norm_params,
                                 **common_args)
    conv_args_relu = make_args(activation=relu, **conv_args_common)
    conv_args_none = make_args(activation=None, **conv_args_common)
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut', **common_args)
        else:
            shortcut = conv2d(inputs, depth, filter_size=(1, 1), stride=(stride, stride),
                              name='shortcut', **conv_args_none)

        residual = conv2d(inputs, depth_bottleneck, filter_size=(1, 1), stride=(1, 1),
                          name='conv1', **conv_args_relu)
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               rate=rate, scope='conv2', **conv_args_relu)
        residual = conv2d(residual, depth, filter_size=(1, 1), stride=(1, 1),
                          name='conv3', **conv_args_none)

        # not in endpoints. does that matter?
        output = tf.nn.relu(shortcut + residual)

        return _collect_named_outputs(common_args['outputs_collections'],
                                      sc.name,
                                      output)


def stack_blocks_dense(net, blocks, output_stride=None,
                       **common_args):
    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit

                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=1,
                                            rate=rate,
                                            **common_args)
                        rate *= unit_stride

                    else:
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=unit_stride,
                                            rate=1,
                                            **common_args)
                        current_stride *= unit_stride
            net = _collect_named_outputs(common_args['outputs_collections'], sc.name, net)
    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              scope=None,
              **common_args):
    conv_args = make_args(use_bias=False, activation=relu, batch_norm=batch_norm_tf, batch_norm_args=batch_norm_params,
                          **common_args)
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=common_args['reuse']) as sc:
        net = inputs
        if include_root_block:
            if output_stride is not None:
                if output_stride % 4 != 0:
                    raise ValueError('The output_stride needs to be a multiple of 4.')
                output_stride /= 4
            net = conv2d_same(net, 64, 7, stride=2, scope='conv1', **conv_args)
            net = max_pool(net, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool1')
        net = stack_blocks_dense(net, blocks, output_stride, **common_args)
        if global_pool:
            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        if num_classes is not None:
            net = conv2d(net, num_classes, filter_size=(1, 1), activation=None,
                         name='logits', **common_args)
            predictions = softmax(net, name='predictions', **common_args)
        return end_points(common_args['is_training'])

