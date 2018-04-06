import tensorflow as tf
import generator
import preprocess
import numpy as np
import functools
import YOLO
import argparse


def yolo_model_fn(features, labels, mode):
    pred = YOLO.forward(features, mode == tf.estimator.ModeKeys.TRAIN)
    loss = YOLO.loss(pred, labels)
    loss = tf.identity(loss, name='loss')

    global_step = tf.train.get_global_step()

    # TODO: add mAP metric
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: add prediction mode
        return None
        # predictions = {
        # }
        # return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        raise Exception('unexpected mode')


def train_input_fn(name2idx, prior_bboxes, batch_size=16):
    gen = functools.partial(generator.data_generator, ".", name2idx, prior_bboxes)
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32),
                                             (
                                                 tf.TensorShape([None, None, 3]),
                                                 tf.TensorShape([None, None, 3, None]),
                                                 tf.TensorShape([None, None, 3, None]),
                                                 tf.TensorShape([None, None, 3, None])
                                             ))
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(1)
    next_batch = dataset.make_one_shot_iterator().get_next()
    return next_batch[0], next_batch[1:]


def eval_input_fn(name2idx, prior_bboxes, batch_size=16):
    gen = functools.partial(generator.data_generator, ".", name2idx, prior_bboxes)
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32),
                                             (
                                                 tf.TensorShape([None, None, 3]),
                                                 tf.TensorShape([None, None, 3, None]),
                                                 tf.TensorShape([None, None, 3, None]),
                                                 tf.TensorShape([None, None, 3, None])
                                             ))
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(1)
    next_batch = dataset.make_one_shot_iterator().get_next()
    return next_batch[0], next_batch[1:]


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    classes, name2idx, idx2name, _ = preprocess.get_classes_and_bboxes()
    prior_bboxes = np.load('prior_bboxes.npy')

    # parse input
    parser = argparse.ArgumentParser(description='Tensorflow Butterfly')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to run')
    args = parser.parse_args()

    # build inputs
    train_inputs = train_input_fn(name2idx, prior_bboxes)
    eval_inputs = eval_input_fn(name2idx, prior_bboxes)

    # build estimator
    estimator = tf.estimator.Estimator(
        model_fn=yolo_model_fn,
        model_dir='models/yolo_model'
    )
    logging_hook = tf.train.LoggingTensorHook(
        tensors={
            "loss": "loss",
        }, every_n_iter=100)

    for i in range(args.epochs):
        estimator.train(input_fn=train_input_fn, steps=1e4, hooks=[logging_hook])
        estimator.evaluate(input_fn=eval_input_fn, steps=1e3, hooks=[logging_hook])

