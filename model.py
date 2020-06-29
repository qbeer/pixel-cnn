import tensorflow as tf
from mask_conv import MaskConv2D
from residual_block import ResidualBlock


class PixelCNN(tf.keras.Model):
    def __init__(self,
                 input_shape=(32, 32, 3),
                 hidden_size=120,
                 n_residual_blocks=10,
                 color_conditioning=True,
                 n_logits=10):

        super(PixelCNN, self).__init__()

        self.convA = MaskConv2D(mask_type='A',
                                color_conditioning=True,
                                filters=2 * hidden_size,
                                kernel_size=(7, 7),
                                activation='relu')

        self.res_blocks = [
            ResidualBlock(hidden_size, color_conditioning)
            for _ in range(n_residual_blocks)
        ]

        self.convB_1 = MaskConv2D(mask_type='B',
                                  color_conditioning=True,
                                  filters=4 * hidden_size,
                                  kernel_size=(1, 1),
                                  activation=None)

        self.convB_2 = MaskConv2D(
            mask_type='B',
            color_conditioning=True,
            filters=3 * 3 *
            n_logits,  # RGB * params for mixture of logistics ( pi_i, mu_i, s_i ) * n_logits
            kernel_size=(1, 1),
            activation=None)

    def call(self, inputs):
        x = self.convA(inputs)
        for res in self.res_blocks:
            x = res(x)
        x = tf.nn.relu(x)
        x = self.convB_1(x)
        x = tf.nn.relu(x)
        x = self.convB_2(x)
        return x
