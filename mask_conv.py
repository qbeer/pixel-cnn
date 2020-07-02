import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn, nn_ops


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
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self._padding_op,
            data_format=self._conv_op_data_format)

        self.mask = self.create_mask(self.mask_type, self.color_conditioning)

        self.built = True

    def call(self, inputs):
        """
            Equivalent to a basic kernel masking. Nothing special. :)
            After that call the original convolutional operation.
        """
        kernel = self.kernel * self.mask

        if self._recreate_conv_op(inputs):
            self._convolution_op = nn_ops.Convolution(
                inputs.get_shape(),
                filter_shape=self.kernel.shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self._padding_op,
                data_format=self._conv_op_data_format)

        outputs = self._convolution_op(inputs, kernel)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def create_mask(self, mask_type, color_conditioning):
        K, _, C_in, C_out = self.kernel.shape
        mask = np.zeros(shape=(K, K, C_in, C_out))
        mask[:K // 2, :, :, :] = 1
        mask[K // 2, :K // 2, :, :] = 1
        if color_conditioning:
            # mapping from e.g. : R, G, B to RRR, GGG, BBB
            assert C_in % 3 == 0 and C_out % 3 == 0, 'Input and output channels must be multiples of 3!'
            C_in_third, C_out_third = C_in // 3, C_out // 3
            if mask_type == 'B':
                mask[K // 2, K // 2, :C_in_third, :
                     C_out_third] = 1  # conditioning the center pixel on R | R
                mask[K // 2, K // 2, :2 * C_in_third, C_out_third:2 *
                     C_out_third] = 1  # -ii- on G | RG
                mask[K // 2, K // 2, :, 2 * C_out_third] = 1  # -ii- on B | RGB
            elif mask_type == 'A':
                """
                    Only used for the first convolution from the RGB input. It shifts the receptive field
                    to the direction of the top-left corner, successive applications would results in no
                    receptive field in deeper layers.
                """
                mask[K // 2, K // 2, :C_in_third, C_out_third:2 *
                     C_out_third] = 1  # conditioning center pixel on G | R
                mask[K // 2, K // 2, :2 * C_in_third, 2 *
                     C_out_third:] = 1  # -ii- on B | RG
        else:
            if mask_type == 'B':
                mask[K // 2, K // 2, :, :] = 1  # condition on center pixel

        return tf.constant(mask, dtype=tf.float32)