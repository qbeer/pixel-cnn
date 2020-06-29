import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow_datasets as tfds
from model import PixelCNN

import numpy as np

tfd = tfp.distributions

ds_train = tfds.load('cifar10',
                     split='train',
                     shuffle_files='True',
                     batch_size=16,
                     as_supervised=True)
ds_test = tfds.load('cifar10',
                    split='test',
                    shuffle_files='False',
                    batch_size=32,
                    as_supervised=True)

model = PixelCNN()


def neg_log_likelihood(target, output, n_mixtures):
    B, H, W, total_channels = output.shape
    assert total_channels == 9 * n_mixtures, 'Total channels should be equal to 9 times the number of mixture models. (RGB + pi, mu, s)'
    output = tf.reshape(output, shape=(B, H, W, 3, 3 * n_mixtures))
    means = output[..., :n_mixtures]
    log_scales_inverse = output[..., n_mixtures:2 * n_mixtures]
    mixture_scales = output[..., n_mixtures * 2:]

    mixture_scales = tf.nn.softmax(mixture_scales, axis=4)  # last index
    scales_inverse = tf.math.exp(log_scales_inverse)

    targets = tf.stack([target for _ in range(n_mixtures)], axis=-1)

    arg_plus = (targets + .5 - means) * scales_inverse
    arg_minus = (targets - .5 - means) * scales_inverse

    normal_cdf = tf.reduce_sum(
        (tf.nn.sigmoid(arg_plus) - tf.nn.sigmoid(arg_minus)) * mixture_scales,
        axis=-1)
    underflow_cdf = tf.reduce_sum(tf.nn.sigmoid(arg_plus) * mixture_scales,
                                  axis=-1)
    overflow_cdf = tf.reduce_sum(
        (1. - tf.nn.sigmoid(arg_minus)) * mixture_scales, axis=-1)

    #print(target)

    probs = tf.where(target < -.99, underflow_cdf,
             tf.where(target > .99, overflow_cdf, normal_cdf))

    log_probs = tf.math.log(probs)

    return tf.reduce_mean(-tf.reduce_sum(log_probs, axis=[1, 2, 3])) # reduce to sum of negative log_likelihood

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

_it = 0

for images, _ in ds_train:
    _it += 1
    images = tf.cast(images, dtype=tf.float32) / 127.5 - 1.
    with tf.GradientTape() as tape:
        outputs = model(images)
        nll = neg_log_likelihood(images, outputs, 10)
    grads = tape.gradient(nll, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    nats = nll.numpy() * np.log2(np.e) / (32 * 32)
    if _it % 100 == 0:
        print(nll.numpy(), nats)

