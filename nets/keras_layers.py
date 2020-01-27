# !/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

import math
import numpy as np

from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers

class SliceLayer(Layer):
    def __init__(self, name, **kwargs):
        self.name = name
        self.index = -1
        super(SliceLayer, self).__init__(**kwargs)

    def set_index(self, index):
        self.index = index

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        return output_shape

    def call(self, input, mask=None):
        value = input[:, self.index, :]
        return value

class ReshapeLayer(Layer):
    def __init__(self, new_shape, **kwargs):
        self.new_shape = new_shape
        super(ReshapeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]] + list(self.new_shape)
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        output_shape = [-1] + list(self.new_shape)
        output_shape = tuple(output_shape)
        value = tf.reshape(input, output_shape)
        return value

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'new_shape': self.new_shape}
        base_config = super(ReshapeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TransposeLayer(Layer):
    def __init__(self, new_perm, **kwargs):
        self.new_perm = new_perm
        super(TransposeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == len(self.new_perm)

        output_shape = [input_shape[self.new_perm[idx]] for idx in range(len(input_shape))]
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        value = tf.transpose(input, self.new_perm)
        return value

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'new_perm': self.new_perm}
        base_config = super(TransposeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(ExpandDimsLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]

        for axis in axes:
            output_shape.insert(axis, 1)

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):

        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        value = input

        for axis in axes:
            value = tf.expand_dims(value, axis)

        return value

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandDimsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(SqueezeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if i not in axes:
                output_shape.append(input_shape[i])
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        value = tf.squeeze(input, self.axis)
        return value

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaxLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(MaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_max(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(MaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SumLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(SumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SumLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_sum(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(SumLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(MeanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_mean(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(MeanLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NetVLAD(Layer):
    """Creates a NetVLAD class.
    """

    def __init__(self, feature_size, max_samples, cluster_size, **kwargs):
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.cluster_size = cluster_size
        super(NetVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.cluster_weights1 = self.add_weight(name='kernel_W1', shape=(self.feature_size, self.cluster_size), initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)), trainable=True)
        self.cluster_biases = self.add_weight(name='kernel_B1', shape=(self.cluster_size,), initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)), trainable=True)
        self.cluster_weights2 = self.add_weight(name='kernel_W2', shape=(1, self.feature_size, self.cluster_size), initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)), trainable=True)

        super(NetVLAD, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        output_shape = tuple([None, self.cluster_size * self.feature_size])
        return output_shape

    def call(self, reshaped_input, mask=None):
        """Forward pass of a NetVLAD block.
        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])
        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul

        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]

        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """
        activation = K.dot(reshaped_input, self.cluster_weights1)

        activation += self.cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        a = tf.multiply(a_sum, self.cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_samples, self.feature_size])

        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)
        # vlad = K.dot(vlad, self.hidden1_weights)

        return vlad

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {'cluster_size': self.cluster_size, 'feature_size': self.feature_size, 'max_samples': self.max_samples}
        base_config = super(NetVLAD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepthwiseConv1DLayer(Layer):
    """
    Expects a tensor of 5D (Batch_Size, Temporal_Dimension, Width, Length, Channel_Dimension)
    Applies a local 1*1*k Conv1D on each separate channel of the input, and along the temporal dimension
    Returns a 5D tensor.
    """

    def __init__(self, kernel_size, padding, **kwargs):
        self.kernel_size = kernel_size
        self.padding = padding
        super(DepthwiseConv1DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 20, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 5

        initializer = contrib_layers.xavier_initializer()

        _, n_timesteps, feat_map_side_dim1, feat_map_side_dim2, n_spatial_maps = input_shape
        self.n_timesteps = n_timesteps
        self.n_maps = n_spatial_maps
        self.side_dim1 = feat_map_side_dim1
        self.side_dim2 = feat_map_side_dim2

        weights_name = 'depthwise_conv_1d_weights'
        biases_name = 'depthwise_conv1d_biases'

        # 1x1 convolution kernel
        weights_shape = [self.kernel_size, 1, n_spatial_maps, 1]
        bias_shape = [n_spatial_maps, ]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(biases_name, shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseConv1DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 20, 7, 7, 1024)

        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = tf.transpose(input, (0, 2, 3, 1, 4))  # (None, 7, 7, 20, 1, 1024)
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, 1, self.n_maps))  # (None*7*7, 20, 1, 1024)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = tf.nn.depthwise_conv2d(tensor, self.conv_weights, strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC')  # (None*7*7, 20, 1, 1024)
        tensor = tf.nn.bias_add(tensor, self.conv_biases)  # (None*7*7, 20, 1, 1024)

        # reshape to get the spatial dimensions
        tensor = tf.reshape(tensor, (-1, self.side_dim1, self.side_dim2, self.n_timesteps, self.n_maps))  # (None, 7, 7, 20, 1024)

        # finally, transpose to get the desired output shape
        tensor = tf.transpose(tensor, (0, 3, 1, 2, 4))  # (None, 20, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'kernel_size': self.kernel_size, 'padding': self.padding}
        base_config = super(DepthwiseConv1DLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepthwiseDilatedConv1DLayer(Layer):
    """
    Expects a tensor of 5D (Batch_Size, Temporal_Dimension, Width, Length, Channel_Dimension)
    Applies a local 1*1*k Conv1D on each separate channel of the input, and along the temporal dimension
    Returns a 5D tensor.
    """

    def __init__(self, kernel_size, dilation_rate, padding, **kwargs):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        super(DepthwiseDilatedConv1DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 20, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 5

        initializer = contrib_layers.xavier_initializer()

        _, n_timesteps, feat_map_side_dim1, feat_map_side_dim2, n_spatial_maps = input_shape
        self.n_timesteps = n_timesteps
        self.n_maps = n_spatial_maps
        self.side_dim1 = feat_map_side_dim1
        self.side_dim2 = feat_map_side_dim2

        weights_name = 'depthwise_conv_1d_weights'
        biases_name = 'depthwise_conv1d_biases'

        # 1x1 convolution kernel
        weights_shape = [self.kernel_size, 1, n_spatial_maps, 1]
        bias_shape = [n_spatial_maps, ]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(biases_name, shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseDilatedConv1DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 20, 7, 7, 1024)

        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = tf.transpose(input, (0, 2, 3, 1, 4))  # (None, 7, 7, 20, 1, 1024)
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, 1, self.n_maps))  # (None*7*7, 20, 1, 1024)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = tf.nn.depthwise_conv2d(tensor, self.conv_weights, strides=(1, 1, 1, 1), rate=(self.dilation_rate, self.dilation_rate), padding='SAME', data_format='NHWC')  # (None*7*7, 20, 1, 1024)
        tensor = tf.nn.bias_add(tensor, self.conv_biases)  # (None*7*7, 20, 1, 1024)

        # reshape to get the spatial dimensions
        tensor = tf.reshape(tensor, (-1, self.side_dim1, self.side_dim2, self.n_timesteps, self.n_maps))  # (None, 7, 7, 20, 1024)

        # finally, transpose to get the desired output shape
        tensor = tf.transpose(tensor, (0, 3, 1, 2, 4))  # (None, 20, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'kernel_size': self.kernel_size, 'dilation_rate': self.dilation_rate, 'padding': self.padding}
        base_config = super(DepthwiseDilatedConv1DLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeceptionTemporalBlock(Layer):
    """
    Given tensor, split it into
    """

    def __init__(self, keras_layers, n_groups, n_channels_per_group_in, n_channels_per_branch, **kwargs):
        self.n_groups = n_groups
        self.keras_layers = keras_layers
        self.n_channels_per_branch = n_channels_per_branch
        self.n_channels_per_group_in = n_channels_per_group_in
        super(TimeceptionTemporalBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, T, 1024)
        :param input_shape:
        :return:
        """

        n_branches = 5
        n_groups = self.n_groups
        n_channels_per_branch = self.n_channels_per_branch
        n_channels_out = n_groups * n_branches * n_channels_per_branch
        self.n_channels_out = n_channels_out

        n_per_group = self.n_channels_per_group_in
        idxes = [(i * n_per_group, (i + 1) * n_per_group) for i in range(self.n_groups)]
        self.idxes = tf.constant(idxes)

        [l_01, l_02, l_31, l_32, l_33, l_51, l_52, l_53, l_71, l_72, l_73, l_11, l_12, l_13] = self.keras_layers

        self.l_01 = l_01
        self.l_02 = l_02

        self.l_31 = l_31
        self.l_32 = l_32
        self.l_33 = l_33

        self.l_51 = l_51
        self.l_52 = l_52
        self.l_53 = l_53

        self.l_71 = l_71
        self.l_72 = l_72
        self.l_73 = l_73

        self.l_11 = l_11
        self.l_12 = l_12
        self.l_13 = l_13

        self.trainable_weights = []
        super(TimeceptionTemporalBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[-1] = self.n_channels_out
        return output_shape

    def call(self, input, mask=None):
        # inputs is of shape (None, T, 1024)

        # self.__input_tensor = input

        input_shape = K.int_shape(input)
        _, n_timesteps, side_dim1, side_dim2, n_channels_in = input_shape

        n_groups = self.n_groups
        n_per_group = self.n_channels_per_group_in

        # split into groups
        tensor = tf.reshape(input, (-1, n_timesteps, side_dim1, side_dim2, n_groups, n_per_group))
        tensor = tf.transpose(tensor, (4, 0, 1, 2, 3, 5))
        self.__input_tensor = tensor

        idxes = np.arange(n_groups)
        idxes = tf.constant(idxes, dtype=tf.int32)

        # apply timeception block for each group
        graph_vector = tf.map_fn(self.__timeception_block, elems=idxes, parallel_iterations=self.n_groups, dtype=tf.float32)  # (T, None, N, N)
        graph_vector = tf.transpose(graph_vector, (1, 0))

        return graph_vector

    def __timeception_block(self, idx):
        input = self.__input_tensor
        tensor_group = input[idx]

        # branch 1: dimension reduction only and no temporal conv
        t_0 = self.l_01(tensor_group)
        t_0 = self.l_02(t_0)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        t_3 = self.l_31(tensor_group)
        t_3 = self.l_32(t_3)
        t_3 = self.l_33(t_3)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        t_5 = self.l_51(tensor_group)
        t_5 = self.l_52(t_5)
        t_5 = self.l_53(t_5)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        t_7 = self.l_71(tensor_group)
        t_7 = self.l_72(t_7)
        t_7 = self.l_73(t_7)

        # branch 5: dimension reduction followed by temporal max pooling
        t_1 = self.l_11(tensor_group)
        t_1 = self.l_12(t_1)
        t_1 = self.l_13(t_1)

        output = tf.concat([t_0, t_3, t_5, t_7, t_1])
        _ = 10
        return output