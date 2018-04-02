import tensorflow as tf
import tensorflow.contrib.slim as slim


def DarkBlock(input, filter_in, filter_out, training, residual=True):
    x = input

    x = slim.batch_norm(x, is_training=training)
    x = slim.conv2d(x, filter_in, 1, activation_fn=None)
    x = tf.nn.relu(x)

    x = slim.batch_norm(x, is_training=training)
    x = slim.conv2d(x, filter_out, 3, activation_fn=None)
    x = tf.nn.relu(x)

    return x + input if residual else x
