import tensorflow as tf
import tensorflow.contrib.slim as slim
from DarkBlock import DarkBlock
from BackBone import Darknet53
import config


def forward(input, training):
    output = []
    x = input
    x, feature_maps = Darknet53.forward(x, training)

    feature = None
    # coarsest scale
    for i in range(3):
        x = DarkBlock(x, 512, 1024, training, False)
        if i == 1:
            feature = x

    output.append(slim.conv2d(x, 3 * (4 + 1 + config.NUM_CLASSES), 1, activation_fn=None))

    x = tf.keras.layers.UpSampling2D()(feature) + feature_maps[0]

    # finer scale
    for i in range(3):
        x = DarkBlock(x, 256, 512, training, False)
        if i == 1:
            feature = x

    output.append(slim.conv2d(x, 3 * (4 + 1 + config.NUM_CLASSES), 1, activation_fn=None))

    x = tf.keras.layers.UpSampling2D()(feature) + feature_maps[1]

    # finest scale
    for i in range(3):
        x = DarkBlock(x, 128, 256, training, False)

    output.append(slim.conv2d(x, 3 * (4 + 1 + config.NUM_CLASSES), 1, activation_fn=None))

    # TODO: add logistic regression for class labels
    return output