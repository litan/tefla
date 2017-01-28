from __future__ import division, print_function, absolute_import

import pytest
import tensorflow as tf

from tefla.core.layers import fully_connected


@pytest.fixture(autouse=True)
def clean_graph():
    tf.reset_default_graph()


def test_trainable_true():
    x = tf.placeholder(tf.float32, [1, 10, 10, 3])
    x = fully_connected(x, 15, is_training=True, reuse=False, name='fc1', trainable=True)
    trainable_vars = [v.name for v in tf.trainable_variables()]
    assert 'fc1/weights:0' in trainable_vars
    assert 'fc1/biases:0' in trainable_vars


def test_trainable_false():
    x = tf.placeholder(tf.float32, [1, 10, 10, 3])
    x = fully_connected(x, 15, is_training=True, reuse=False, name='fc1', trainable=False)
    all_vars = set(tf.global_variables())
    trainable_vars = set(tf.trainable_variables())
    non_trainable_vars = all_vars.difference(trainable_vars)
    trainable_vars = [v.name for v in trainable_vars]
    non_trainable_vars = [v.name for v in non_trainable_vars]
    assert 'fc1/weights:0' not in trainable_vars
    assert 'fc1/biases:0' not in trainable_vars
    assert 'fc1/weights:0' in non_trainable_vars
    assert 'fc1/biases:0' in non_trainable_vars


if __name__ == '__main__':
    pytest.main([__file__])
