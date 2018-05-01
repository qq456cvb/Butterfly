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


def get_pred_bbox(preds, prior_bboxes):
    scales = [32, 16, 8]
    bboxes = []
    classes = []
    for i in range(3):
        N = config.IMG_SIZE // scales[i]
        pred = preds[i]
        # B * 4
        valid_idx = tf.where(tf.greater(tf.sigmoid(pred[:, :, :, :, 4]), 0.5))

        # B * (5 + 94)
        valid_pred = tf.gather_nd(pred, valid_idx)
        # three bbox in each level
        bboxes_level = []
        classes_level = []
        for j in range(3):
            bbox_idx = tf.where(tf.equal(valid_idx[:, 3], j))
            valid_bbox_idx = tf.gather_nd(valid_idx, bbox_idx)
            bbox_pred = tf.gather_nd(valid_pred, bbox_idx)
            bbox_lt = (tf.cast(valid_bbox_idx[:, 1:3], tf.float32) + tf.sigmoid(bbox_pred[:, :2])) / N
            bbox_br = bbox_lt + prior_bboxes[i * 3 + j:i * 3 + j + 1, :] / config.IMG_SIZE * tf.exp(bbox_pred[:, 2:4])
            bbox_cls = tf.expand_dims(tf.argmax(tf.sigmoid(bbox_pred[:, 5:]), axis=1), -1)
            bboxes_level.append(tf.concat([bbox_lt, bbox_br], axis=1))
            classes_level.append(bbox_cls)
        bboxes_level = tf.concat(bboxes_level, axis=0)
        classes_level = tf.concat(classes_level, axis=0)
        bboxes.append(bboxes_level)
        classes.append(classes_level)
    return tf.concat(bboxes, axis=0), tf.squeeze(tf.concat(classes, axis=0), axis=-1)

    # # nms
    # selected_indices = tf.image.non_max_suppression(bboxes[:, :4], bboxes[:, 4], tf.constant(10, dtype=tf.int32))
    # return tf.gather(bboxes, selected_indices)

