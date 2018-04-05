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
        x, inter = DarkBlock(x, 512, 1024, training, False)
        if i == 1:
            feature = inter

    # IMG_SIZE // 32
    o = slim.conv2d(x, 3 * (4 + 1 + config.NUM_CLASSES), 1, activation_fn=None)
    o = tf.reshape(o, [-1, config.IMG_SIZE // 32, config.IMG_SIZE // 32, 3, 4 + 1 + config.NUM_CLASSES])

    output.append(o)

    x = tf.keras.layers.UpSampling2D()(feature) + feature_maps[1]

    # finer scale
    for i in range(3):
        x, inter = DarkBlock(x, 256, 512, training, False)
        if i == 1:
            feature = inter

    # IMG_SIZE // 16
    o = slim.conv2d(x, 3 * (4 + 1 + config.NUM_CLASSES), 1, activation_fn=None)
    o = tf.reshape(o, [-1, config.IMG_SIZE // 16, config.IMG_SIZE // 16, 3, 4 + 1 + config.NUM_CLASSES])

    output.append(o)

    x = tf.keras.layers.UpSampling2D()(feature) + feature_maps[0]

    # finest scale
    for i in range(3):
        x, _ = DarkBlock(x, 128, 256, training, False)

    # IMG_SIZE // 8
    o = slim.conv2d(x, 3 * (4 + 1 + config.NUM_CLASSES), 1, activation_fn=None)
    o = tf.reshape(o, [-1, config.IMG_SIZE // 8, config.IMG_SIZE // 8, 3, 4 + 1 + config.NUM_CLASSES])

    output.append(o)

    # return logits
    return output


# pred and target are lists
def loss(preds, targets):
    costs = []
    for i in range(3):
        pred = preds[i]
        target = targets[i]

        obj_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=target[:, :, :, :, 4], logits=pred[:, :, :, :, 4]))
        valid_idx = tf.where(tf.equal(target[:, :, :, :, 4], 1))
        valid_pred = tf.gather_nd(pred, valid_idx)
        valid_target = tf.gather_nd(target, valid_idx)
        bbox_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_target[:, :2], logits=valid_pred[:, :2])) + \
            tf.nn.l2_loss(valid_target[:, 2:4] - valid_pred[:, 2:4])
        cate_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_target[:, 5:], logits=valid_pred[:, 5:]))
        cost = obj_cost + bbox_cost + cate_cost
        costs.append(cost)
    return tf.add_n(costs)
