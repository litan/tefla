from __future__ import division, print_function, absolute_import

from tefla.core.layer_arg_ops import common_layer_args, end_points
from tefla.core.layers import dropout, relu
from tefla.core.layers import input, fully_connected, softmax

# sizes - (width, height)
image_size = (224, 224)
crop_size = (224, 224)


# A model that works on top of features extracted from layer vgg_16/conv5/maxpool5 of vgg16
# These features can be extracted using tefla.predict with the --output_layer option
def model(is_training, reuse):
    common_args = common_layer_args(is_training, reuse)

    x = input((None, 7, 7, 512), **common_args)
    # x = batch_norm_tf(x, **common_args)
    x = fully_connected(x, 512 , activation=relu, name='fc1', **common_args)
    x = dropout(x, drop_p=0.5, name='dropout1', **common_args)
    logits = fully_connected(x, 6, activation=None, name='logits', **common_args)
    predictions = softmax(logits, name='predictions', **common_args)
    return end_points(is_training)
