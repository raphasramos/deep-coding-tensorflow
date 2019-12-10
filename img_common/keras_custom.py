""" File containing custom operations and layers for keras models """

import tensorflow as tf
import numpy as np


# TODO: give the option to work with {-1, 1} and {0, 1} codes
class ConvBin(tf.keras.layers.Layer):
    """
    Keras layer that implements binarization as implemented in
    https://github.com/1zb/pytorch-image-comp-rnn. It's supposed to implement
    Toderici's article, but there's no stochasticity as proposed in that work.
    This commit follow the implementation.
    """
    def __init__(self, **kwargs):
        if ('activation' in kwargs) and (kwargs['activation'] != 'tanh'):
            raise ValueError('Only tanh is currently supported')
        kwargs['use_bias'] = False
        self.conv = tf.keras.layers.Conv2D(**kwargs)
        super(ConvBin, self).__init__(name=kwargs['name'])

    def get_config(self):
        config = self.conv.get_config()
        return config

    def build(self, input_shape):
        super(ConvBin, self).build(input_shape)

    def call(self, inputs, training=False, *args, **kwargs):
        tensor = self.conv(inputs)
        tensor = self._tf_sign(tensor, training)
        return tensor

    @staticmethod
    @tf.custom_gradient
    def _tf_sign(tensor, training):
        """
        Non-differentiable function used in the context of this layer with
        identity backpropagation gradients.
        """
        def grad(dy):
            return dy, None

        out = tf.numpy_function(ConvBin._prob_sign, [tensor, training],
                                [tf.float32])
        if not tf.executing_eagerly():
            out = out[0]
            out.set_shape(tensor.shape)

        return out, grad

    @staticmethod
    def _prob_sign(in_array, training):
        """
        Function used with numpy_function. It's necessary to use it easily.
        Keras still builds a graph before, so if it's defined there, it
        would have to use graph-based approach.
        """
        if training:
            prob = np.random.uniform(size=in_array.shape)
            out_array = np.copy(in_array)
            out_array[(1 - in_array) / 2 <= prob] = 1
            out_array[(1 - in_array) / 2 > prob] = -1
            return out_array

        return np.sign(in_array)


class Quantize(tf.keras.layers.Layer):
    """
    Keras layer representing quantization. In training time it
    simply puts random noise on data. On test time it's made a simple
    round operation.
    """
    def __init__(self, factor=1, name=None):
        self.factor = tf.constant(factor, dtype=tf.float32)
        self.half = tf.constant(.5, dtype=tf.float32)
        super(Quantize, self).__init__(name=name)

    def build(self, input_shape):
        super(Quantize, self).build(input_shape)

    def call(self, inputs):
        tensor, training = inputs
        tensor *= self.factor
        if not training:
            output = tf.math.round(tensor)
        else:
            noise = tf.random.uniform(tf.shape(tensor), -self.half,
                                      self.half)
            output = tensor + noise

        return output

    def compute_output_shape(self, input_shape):
        tensor_shape, flag_shape = input_shape
        return tensor_shape


class DepthToSpace(tf.keras.layers.Layer):
    """
    Keras layer implementation for depth to space operation of tensorflow.
    """
    def __init__(self, block_size, data_format='NHWC', name=None):
        super(DepthToSpace, self).__init__(name=name)
        self.block_size = block_size
        self.data_format = data_format
        self.depth_to_space = None

    def get_config(self):
        config = {'block_size': self.block_size,
                  'data_format': self.data_format,
                  "name": self.name}
        return config

    def build(self, input_shape):
        self.depth_to_space = lambda tensor: \
            tf.nn.depth_to_space(tensor, block_size=self.block_size,
                                 data_format=self.data_format)
        super(DepthToSpace, self).build(input_shape)

    def call(self, tensor):
        output = self.depth_to_space(tensor)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = np.array(input_shape.as_list())
        output_shape[1:3] *= self.block_size
        output_shape[3] /= (2 * self.block_size)
        return output_shape


class CustomConv2DLSTM(tf.keras.layers.Layer):
    """
    Keras layer implementation for a LSTM which does not account for the
    time component.
    """
    def __init__(self, filters, kernel_size=3, strides=1, padding='same',
                 use_bias=True, h_kernel_size=1, name=None):
        self.filters = filters
        self.kwargs = {'filters': 4 * self.filters, 'kernel_size': kernel_size,
                       'strides': strides, 'padding': padding,
                       'use_bias': use_bias}
        self.h_kwargs = {'filters': 4 * self.filters,
                         'kernel_size': h_kernel_size, 'strides': 1,
                         'padding': padding, 'use_bias': use_bias}
        super(CustomConv2DLSTM, self).__init__(name=name)

    def get_config(self):
        config = self.kwargs.copy()
        config['filters'] = self.filters
        config['h_kernel_size'] = self.h_kwargs['kernel_size']
        config['name'] = self.name
        return config

    def reset_states(self):
        self.cell_state = tf.zeros_like(self.cell_state)
        self.hidden_state = tf.zeros_like(self.cell_state)

    def build(self, input_shape):
        self.input_conv = tf.keras.layers.Conv2D(**self.kwargs)
        self.hidden_conv = tf.keras.layers.Conv2D(**self.h_kwargs)

        # Construct the hidden and cell states
        out_tensor = self.input_conv(tf.zeros(input_shape))
        aux_gate = tf.split(out_tensor, 4, axis=3)[0]
        self.hidden_state = tf.zeros_like(aux_gate)
        self.cell_state = tf.zeros_like(aux_gate)

        self.stateful = True

        super(CustomConv2DLSTM, self).build(input_shape)

    def call(self, input):
        gates = self.input_conv(input) + self.hidden_conv(self.hidden_state)
        input_g, forget_g, cell_g, output_g = tf.split(gates, 4, axis=3)

        input_g = tf.math.sigmoid(input_g)
        forget_g = tf.math.sigmoid(forget_g)
        cell_g = tf.math.tanh(cell_g)
        output_g = tf.math.sigmoid(output_g)

        if tf.executing_eagerly():
            # To avoid static tensors in object variables
            self.cell_state = (forget_g * self.cell_state) + (input_g * cell_g)
            self.hidden_state = output_g * tf.math.tanh(self.cell_state)

        return self.hidden_state