import tensorflow as tf
import generator
import preprocess
import numpy as np
import functools
import YOLO


def rpn_model_fn(features, labels, mode):
    pred = YOLO.forward(features, True)
    return pred, labels, YOLO.loss(pred, labels)


def train_input_fn(name2idx, prior_bboxes):
    gen = functools.partial(generator.data_generator, name2idx, prior_bboxes)
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32),
                                             (
                                                 tf.TensorShape([None, None, 3]),
                                                 tf.TensorShape([None, None, 3, None]),
                                                 tf.TensorShape([None, None, 3, None]),
                                                 tf.TensorShape([None, None, 3, None])
                                             ))
    dataset = dataset.batch(2)
    # dataset = dataset.prefetch(1)
    next_batch = dataset.make_one_shot_iterator().get_next()
    return next_batch[0], next_batch[1:]


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    classes, name2idx, _, _ = preprocess.get_classes_and_bboxes()
    prior_bboxes = np.load('prior_bboxes.npy')
    inputs = train_input_fn(name2idx, prior_bboxes)
    output = rpn_model_fn(inputs[0], inputs[1], 0)
    with tf.train.MonitoredSession() as sess:
        pred, out, test = sess.run(output)
        print(out[0].shape)
        print(out[1].shape)
        print(out[2].shape)
        print(pred[0].shape)
        print(pred[1].shape)
        print(pred[2].shape)
        print(test)

