from model import PixelCNN
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

import tensorflow_probability as tfp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    hyperparams = {
        "mnist": {
            "input_shape": (28, 28, 1),
            "color_conditioning": False,
            "n_mixtures": 5,
            "epochs": 1
        },
        "cifar10": {
            "input_shape": (32, 32, 3),
            "color_conditioning": True,
            "n_mixtures": 10,
            "epochs": 5
        }
    }

    n_mixtures = hyperparams[args.dataset]['n_mixtures']
    color_conditioning = hyperparams[args.dataset]['color_conditioning']
    input_shape = hyperparams[args.dataset]['input_shape']
    epochs = hyperparams[args.dataset]['epochs']

    model = PixelCNN(n_mixtures=n_mixtures,
                     color_conditioning=color_conditioning,
                     input_shape=input_shape)
    model.build(input_shape=(16, *input_shape))
    model.load_weights(f'pixel_cnn_{args.dataset}_25.h5')

    random_input = np.random.uniform(size=(16, *input_shape), low=-1,
                                     high=1).astype(np.float32)

    output = model(random_input)

    for x in range(input_shape[0]):
        for y in range(input_shape[1]):
            for c in range(input_shape[-1]):
                B, H, W, total_channels = output.shape
                output = output = tf.reshape(output,
                                             shape=(B, H, W, input_shape[-1],
                                                    3 * n_mixtures))
                means = output[:, x, y, c, :n_mixtures]
                log_scales_inverse = output[:, x, y, c, n_mixtures:2 *
                                            n_mixtures]
                mixture_scales = output[:, x, y, c, n_mixtures * 2:]

                mixture_scales = tf.nn.softmax(mixture_scales,
                                               axis=-1)  # last index
                scales_inverse = tf.math.exp(log_scales_inverse)

                logistcs = tfp.distributions.Logistic(loc=means,
                                                      scale=1. /
                                                      (scales_inverse + 1e-12))

                sample = tf.reduce_sum(logistcs.sample() * mixture_scales,
                                       axis=-1)

                print(sample)

                random_input[:, x, y, c] = sample

                output = model(random_input)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    random_input = np.clip(random_input, -1, 1)

    for ind, ax in enumerate(axes.flatten()):
        im = (127.5 * (random_input[ind, ...] + 1)).astype(int)
        ax.imshow(
            im.reshape(*input_shape) if input_shape[-1] != 1 else im.
            reshape(*input_shape[:2]))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(f'{args.dataset}_samples_pixelcnn.png')