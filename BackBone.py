import tensorflow as tf
import tensorflow.contrib.slim as slim
from DarkBlock import DarkBlock


class Darknet53:
    # TODO: read structure from file
    @staticmethod
    def forward(input, training):
        # for detecting different scales
        feature_maps = []
        x = input
        x = slim.conv2d(x, 32, 3)
        x = slim.conv2d(x, 64, 3, 2)

        for _ in range(1):
            x = DarkBlock(x, 32, 64, training)

        x = slim.conv2d(x, 128, 3, 2)

        for _ in range(2):
            x = DarkBlock(x, 64, 128, training)

        x = slim.conv2d(x, 256, 3, 2)

        for _ in range(8):
            x = DarkBlock(x, 128, 256, training)

        feature_maps.append(x)

        x = slim.conv2d(x, 512, 3, 2)

        for _ in range(8):
            x = DarkBlock(x, 256, 512, training)

        feature_maps.append(x)

        x = slim.conv2d(x, 1024, 3, 2)

        for _ in range(4):
            x = DarkBlock(x, 512, 1024, training)

        return x, feature_maps
