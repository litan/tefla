# Ported from https://github.com/tensorflow/tensorflow/tree/r0.12/tensorflow/contrib/slim/python/slim/nets
from __future__ import division, print_function

from resnet_common import *


def resnet_v1_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None,
                 scope='resnet_v1_50',
                 **common_args):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v1(inputs, blocks, num_classes,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, scope=scope, **common_args)


# sizes - (width, height)
image_size = (224, 224)
crop_size = (224, 224)


def model(is_training, reuse):
    common_args = common_layer_args(is_training, reuse)
    x = input((None, crop_size[1], crop_size[0], 3), **common_args)
    model_name = 'resnet_v1_50'
    mean_rgb = tf.get_variable(name='%s/mean_rgb' % model_name, initializer=tf.truncated_normal(shape=[3]),
                               trainable=False)
    x = x - mean_rgb
    resnet = resnet_v1_50(x, 1000, scope=model_name, **common_args)
    resnet['predictions'] = resnet['%s/predictions' % model_name]
    return resnet

# resnet_model = model(False, None)
# v = tf.get_variable('dummy', [5])
# from tefla.utils import util
# util.show_vars()
# weights_from = '/media/lalit/work/tefla-weightloading/resnet_v1_50.ckpt'
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     util.load_variables(sess, saver, weights_from)
