import tensorflow as tf
from mask_conv import MaskConv2D


class ResidualBlock(tf.keras.Model):
    def __init__(self, hidden_size, color_conditioning):
        super(ResidualBlock, self).__init__()
        self.convB_1 = MaskConv2D(mask_type='B',
                                  color_conditioning=color_conditioning,
                                  filters=hidden_size,
                                  kernel_size=(1, 1),
                                  activation=None)
        self.convB_2 = MaskConv2D(mask_type='B',
                                  color_conditioning=color_conditioning,
                                  filters=hidden_size,
                                  kernel_size=(3, 3),
                                  activation=None)
        self.convB_3 = MaskConv2D(mask_type='B',
                                  color_conditioning=color_conditioning,
                                  filters=2 * hidden_size,
                                  kernel_size=(1, 1),
                                  activation=None)

    def call(self, inputs):
        x = tf.nn.relu(inputs)
        x = self.convB_1(x)
        x = tf.nn.relu(x)
        x = self.convB_2(x)
        x = tf.nn.relu(x)
        x = self.convB_3(x)
        return tf.math.add(x, inputs)
