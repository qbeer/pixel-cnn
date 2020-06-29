import tensorflow as tf
import numpy as np


class MaskConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 mask_type,
                 color_conditioning,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
            Using padding='same' to keep the resolution and use_bias='False' to remove conditioning on that,
            later on can be used for conditional PixelCNN
        """
        super(MaskConv2D,
              self).__init__(filters, kernel_size, strides, padding,
                             data_format, dilation_rate, activation, use_bias,
                             kernel_initializer, bias_initializer,
                             kernel_regularizer, bias_regularizer,
                             activity_regularizer, kernel_constraint,
                             bias_constraint, **kwargs)
        self.mask_type = mask_type
        self.color_conditioning = color_conditioning

    def build(self, input_shape):
        super().build(input_shape)
        self.mask = np.zeros(
            shape=(self.kernel.shape),
            dtype=np.float32)  # kernel shape = (K, K, C_in, C_out)
        self.create_mask(self.mask_type, self.color_conditioning)

    def create_mask(self, mask_type, color_conditioning):
        K, _, C_in, C_out = self.kernel.shape
        self.mask[:K // 2, :, :, :] = 1
        self.mask[K // 2, :K // 2, :, :] = 1
        # mapping from e.g. : R, G, B to RRR, GGG, BBB
        assert C_in % 3 == 0 and C_out % 3 == 0, 'Input and output channels must be multiples of 3!'
        if color_conditioning:
            C_in_third, C_out_third = C_in // 3, C_out // 3
            if mask_type == 'B':
                self.mask[
                    K // 2, K // 2, :C_in_third, :
                    C_out_third] = 1  # conditioning the center pixel on R | R
                self.mask[K // 2, K // 2, :2 * C_in_third, C_out_third:2 *
                          C_out_third] = 1  # -ii- on G | RG
                self.mask[K // 2, K // 2, :, 2 *
                          C_out_third] = 1  # -ii- on B | RGB
            elif mask_type == 'A':
                """
                    Only used for the first convolution from the RGB input. It shifts the receptive field
                    to the direction of the top-left corner, successive applications would results in no
                    receptive field in deeper layers.
                """
                self.mask[
                    K // 2, K // 2, :C_in_third, C_out_third:2 *
                    C_out_third] = 1  # conditioning center pixel on G | R
                self.mask[K // 2, K // 2, :2 * C_in_third, 2 *
                          C_out_third:] = 1  # -ii- on B | RG
        else:
            if mask_type == 'B':
                self.mask[K // 2, K //
                          2, :, :] = 1  # condition on center pixel

        self.mask = tf.convert_to_tensor(self.mask)

    def call(self, inputs):
        """
            Equivalent to a basic kernel masking. Nothing special. :)
            After that call the original convolutional operation.
        """
        self.kernel = self.kernel * self.mask
        return super().call(inputs)
