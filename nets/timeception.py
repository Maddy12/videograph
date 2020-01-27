import copy

import tensorflow as tf

import keras.backend as K
from keras import layers as layer_module
from keras.models import Model
from keras.layers import Concatenate, BatchNormalization, Activation, Lambda, MaxPooling3D, Conv3D

from nets.keras_layers import ReshapeLayer, TransposeLayer, DepthwiseDilatedConv1DLayer, DepthwiseConv1DLayer
from nets.keras_layers import MaxLayer, MeanLayer, SumLayer, ExpandDimsLayer, SqueezeLayer, TimeceptionTemporalBlock

# region Timeception Creator

def timeception_temporal_convolutions(tensor, n_layers, n_groups, expansion_factor, is_dilated=True):
    input_shape = K.int_shape(tensor)
    assert len(input_shape) == 5

    _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape

    # collapse regions in one dim
    tensor = ReshapeLayer((n_timesteps, side_dim * side_dim, 1, n_channels_in))(tensor)

    for i in range(n_layers):

        n_channels_per_branch, n_channels_out = __get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

        # add global pooling as local regions
        tensor = __global_spatial_pooling(tensor)

        # temporal conv (inception-style, shuffled)
        if is_dilated:
            tensor = __timeception_shuffled_depthwise_dilated(tensor, n_groups, n_channels_per_branch)
        else:
            tensor = __timeception_shuffled_depthwise(tensor, n_groups, n_channels_per_branch)

        # downsample over time
        tensor = MaxPooling3D(pool_size=(2, 1, 1))(tensor)
        n_channels_in = n_channels_out

    return tensor

def timeception_temporal_convolutions_parallelized(tensor, n_layers, n_groups, expansion_factor, is_dilated=True):
    input_shape = K.int_shape(tensor)
    assert len(input_shape) == 5

    raise Exception('Sorry, not implemented now')

    _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape

    # collapse regions in one dim
    tensor = ReshapeLayer((n_timesteps, side_dim * side_dim, 1, n_channels_in))(tensor)

    for i in range(n_layers):
        # add global pooling as regions
        tensor = __global_spatial_pooling(tensor)

        # temporal conv (inception-style, shuffled)
        n_channels_per_branch, n_channels_out = __get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)
        if is_dilated:
            tensor = __timeception_shuffled_depthwise_dilated_parallelized(tensor, n_groups, n_channels_per_branch)
        else:
            tensor = __timeception_shuffled_depthwise_parallelized(tensor, n_groups, n_channels_per_branch)
        tensor = MaxPooling3D(pool_size=(2, 1, 1))(tensor)
        n_channels_in = n_channels_out

    return tensor

# endregion

# region Timeception Block

def __timeception_shuffled_depthwise(tensor_input, n_groups, n_channels_per_branch):
    _, n_timesteps, side_dim1, side_dim2, n_channels_in = tensor_input.get_shape().as_list()
    assert n_channels_in % n_groups == 0
    n_branches = 5

    n_channels_per_group_in = n_channels_in / n_groups
    n_channels_out = n_groups * n_branches * n_channels_per_branch
    n_channels_per_group_out = n_channels_out / n_groups

    assert n_channels_out % n_groups == 0

    # slice maps into groups
    tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)])(tensor_input)

    t_outputs = []
    for idx_group in range(n_groups):
        tensor_group = tensors[idx_group]

        # branch 1: dimension reduction only and no temporal conv
        t_0 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_0 = BatchNormalization()(t_0)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        t_3 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_3 = DepthwiseConv1DLayer(3, padding='same')(t_3)
        t_3 = BatchNormalization()(t_3)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        t_5 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_5 = DepthwiseConv1DLayer(5, padding='same')(t_5)
        t_5 = BatchNormalization()(t_5)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        t_7 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_7 = DepthwiseConv1DLayer(7, padding='same')(t_7)
        t_7 = BatchNormalization()(t_7)

        # branch 5: dimension reduction followed by temporal max pooling
        t_1 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_1 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same')(t_1)
        t_1 = BatchNormalization()(t_1)

        # concatenate channels of branches
        tensor = Concatenate(axis=4)([t_0, t_3, t_5, t_7, t_1])
        t_outputs.append(tensor)

    # concatenate channels of groups
    tensor = Concatenate(axis=4)(t_outputs)
    tensor = Activation('relu')(tensor)

    # shuffle channels
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group_out))(tensor)
    tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels_out))(tensor)

    return tensor

def __timeception_shuffled_depthwise_dilated(tensor_input, n_groups, n_channels_per_branch):
    _, n_timesteps, side_dim1, side_dim2, n_channels_in = tensor_input.get_shape().as_list()
    assert n_channels_in % n_groups == 0
    n_branches = 5

    n_channels_per_group_in = n_channels_in / n_groups
    n_channels_out = n_groups * n_branches * n_channels_per_branch
    n_channels_per_group_out = n_channels_out / n_groups

    assert n_channels_out % n_groups == 0

    # slice maps into groups
    tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)])(tensor_input)

    t_outputs = []
    for idx_group in range(n_groups):
        tensor_group = tensors[idx_group]

        # branch 1: dimension reduction only and no temporal conv
        t_0 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_0 = BatchNormalization()(t_0)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 1)
        t_3 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_3 = DepthwiseDilatedConv1DLayer(3, 1, padding='same')(t_3)
        t_3 = BatchNormalization()(t_3)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 2)
        t_5 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_5 = DepthwiseDilatedConv1DLayer(3, 2, padding='same')(t_5)
        t_5 = BatchNormalization()(t_5)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 3)
        t_7 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_7 = DepthwiseDilatedConv1DLayer(3, 3, padding='same')(t_7)
        t_7 = BatchNormalization()(t_7)

        # # branch 5: dimension reduction followed by temporal max pooling
        t_1 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_1 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same')(t_1)
        t_1 = BatchNormalization()(t_1)

        # concatenate channels of branches
        tensor = Concatenate(axis=4)([t_0, t_3, t_5, t_7, t_1])
        t_outputs.append(tensor)

    # concatenate channels of groups
    tensor = Concatenate(axis=4)(t_outputs)
    tensor = Activation('relu')(tensor)

    # shuffle channels
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group_out))(tensor)
    tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels_out))(tensor)

    return tensor

# endregion

# region Timeception Block, Parallelized

def __timeception_shuffled_depthwise_parallelized(tensor_input, n_groups, n_channels_per_branch):
    _, n_timesteps, side_dim1, side_dim2, n_channels_in = tensor_input.get_shape().as_list()
    assert n_channels_in % n_groups == 0
    n_branches = 5

    n_channels_per_group_in = n_channels_in / n_groups
    n_channels_out = n_groups * n_branches * n_channels_per_branch
    n_channels_per_group_out = n_channels_out / n_groups

    assert n_channels_out % n_groups == 0

    # define all layers
    l_01 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')
    l_02 = BatchNormalization()

    l_31 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')
    l_32 = DepthwiseConv1DLayer(3, padding='same')
    l_33 = BatchNormalization()

    l_51 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')
    l_52 = DepthwiseConv1DLayer(5, padding='same')
    l_53 = BatchNormalization()

    l_71 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')
    l_72 = DepthwiseConv1DLayer(7, padding='same')
    l_73 = BatchNormalization()

    l_11 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')
    l_12 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same')
    l_13 = BatchNormalization()

    keras_layers = [l_01, l_02, l_31, l_32, l_33, l_51, l_52, l_53, l_71, l_72, l_73, l_11, l_12, l_13]

    tensor = TimeceptionTemporalBlock(keras_layers, n_groups, n_channels_per_group_in,n_channels_per_branch)(tensor_input)

    # slice maps into groups
    tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)])(tensor_input)

    t_outputs = []
    for idx_group in range(n_groups):
        tensor_group = tensors[idx_group]

        # branch 1: dimension reduction only and no temporal conv
        t_0 = l_01(tensor_group)
        t_0 = l_02(t_0)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        t_3 = l_31(tensor_group)
        t_3 = l_32(t_3)
        t_3 = l_33(t_3)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        t_5 = l_51(tensor_group)
        t_5 = l_52(t_5)
        t_5 = l_53(t_5)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        t_7 = l_71(tensor_group)
        t_7 = l_72(t_7)
        t_7 = l_73(t_7)

        # branch 5: dimension reduction followed by temporal max pooling
        t_1 = l_11(tensor_group)
        t_1 = l_12(t_1)
        t_1 = l_13(t_1)

        # concatenate channels of branches
        tensor = Concatenate(axis=4)([t_0, t_3, t_5, t_7, t_1])
        t_outputs.append(tensor)

    # concatenate channels of groups
    tensor = Concatenate(axis=4)(t_outputs)
    tensor = Activation('relu')(tensor)

    # shuffle channels
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group_out))(tensor)
    tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels_out))(tensor)

    return tensor

def __timeception_shuffled_depthwise_dilated_parallelized(tensor_input, n_groups, n_channels_per_branch):
    _, n_timesteps, side_dim1, side_dim2, n_channels_in = tensor_input.get_shape().as_list()
    assert n_channels_in % n_groups == 0
    n_branches = 5

    n_channels_per_group_in = n_channels_in / n_groups
    n_channels_out = n_groups * n_branches * n_channels_per_branch
    n_channels_per_group_out = n_channels_out / n_groups

    assert n_channels_out % n_groups == 0

    # slice maps into groups
    tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)])(tensor_input)

    t_outputs = []
    for idx_group in range(n_groups):
        tensor_group = tensors[idx_group]

        # branch 1: dimension reduction only and no temporal conv
        t_0 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_0 = BatchNormalization()(t_0)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 1)
        t_3 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_3 = DepthwiseDilatedConv1DLayer(3, 1, padding='same')(t_3)
        t_3 = BatchNormalization()(t_3)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 2)
        t_5 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_5 = DepthwiseDilatedConv1DLayer(3, 2, padding='same')(t_5)
        t_5 = BatchNormalization()(t_5)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 3)
        t_7 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_7 = DepthwiseDilatedConv1DLayer(3, 3, padding='same')(t_7)
        t_7 = BatchNormalization()(t_7)

        # # branch 5: dimension reduction followed by temporal max pooling
        t_1 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
        t_1 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same')(t_1)
        t_1 = BatchNormalization()(t_1)

        # concatenate channels of branches
        tensor = Concatenate(axis=4)([t_0, t_3, t_5, t_7, t_1])
        t_outputs.append(tensor)

    # concatenate channels of groups
    tensor = Concatenate(axis=4)(t_outputs)
    tensor = Activation('relu')(tensor)

    # shuffle channels
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group_out))(tensor)
    tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels_out))(tensor)

    return tensor

# endregion

# region Utils

def __global_spatial_pooling(tensor):
    tensor_shape = tensor.get_shape().as_list()
    assert len(tensor_shape) == 5

    # avg pool and max pool, and concat them to spatial dimension
    tensor_ap = MaxLayer(axis=2, is_keep_dim=True)(tensor)
    tensor_mp = MeanLayer(axis=2, is_keep_dim=True)(tensor)
    tensor = Concatenate(axis=2)([tensor, tensor_ap, tensor_mp])
    return tensor

def __get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in):
    n_branches = 5
    n_channels_per_branch = n_channels_in * expansion_factor / float(n_branches * n_groups)
    n_channels_per_branch = int(n_channels_per_branch)
    n_channels_out = n_channels_per_branch * n_groups * n_branches
    n_channels_out = int(n_channels_out)
    return n_channels_per_branch, n_channels_out

def __shuffle_channels(tensor, n_groups):
    tensor_shape = K.int_shape(tensor)
    n_timesteps, side_dim1, side_dim2, n_channels = tensor_shape
    n_channels_per_group = n_channels / n_groups

    assert n_channels_per_group * n_groups == n_channels

    # shuffle channels
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group))(tensor)
    tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
    tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels))(tensor)

    return tensor

# endregion

# region Modules

class Timeception(Model):

    def __init__(self, n_layers, n_groups, expansion_factor, is_dilated=False, name=None):
        self.n_layers = n_layers
        self.n_groups = n_groups
        self.expansion_factor = expansion_factor
        self.is_dilated = is_dilated

        super(Timeception, self).__init__(name=name)

    def compute_output_shape(self, input_shape):
        n_layers = self.n_layers
        expansion_factor = self.expansion_factor
        _, n_timesteps, side_dim_1, side_dim_2, n_channels_in = input_shape

        side_dim = side_dim_1 + side_dim_2
        n_channels_out = n_channels_in

        for l in range(n_layers):
            side_dim += 2
            n_timesteps /= 2
            n_channels_out = int(expansion_factor * n_channels_out)

        output_shape = (None, n_timesteps, side_dim, 1, n_channels_out)
        return output_shape

    def call(self, input, mask=None):

        n_layers = self.n_layers
        is_dilated = self.is_dilated

        _, n_timesteps, side_dim, side_dim, n_channels_in = K.int_shape(input)

        # collapse regions in one dim
        tensor = ReshapeLayer((n_timesteps, side_dim * side_dim, 1, n_channels_in))(input)

        # create n laters
        for i in range(n_layers):

            # add global pooling as regions
            tensor = self.__global_spatial_pooling(tensor)

            # temporal conv (inception-style, shuffled)
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_channels_in)
            if is_dilated:
                tensor = self.__inception_style_temporal_layer_shuffled_depthwise_dilated_complicated(tensor, n_channels_per_branch)
            else:
                tensor = self.__inception_style_temporal_layer_shuffled_depthwise_complicated(tensor, n_channels_per_branch)

            # temporal max pool
            tensor = MaxPooling3D(pool_size=(2, 1, 1))(tensor)
            n_channels_in = n_channels_out

        return tensor

    def __inception_style_temporal_layer_shuffled_depthwise_complicated(self, tensor_input, n_channels_per_branch):
        n_groups = self.n_groups
        _, n_timesteps, side_dim1, side_dim2, n_channels_in = K.int_shape(tensor_input)
        assert n_channels_in % n_groups == 0
        n_branches = 5

        n_channels_per_group_in = n_channels_in / n_groups
        n_channels_out = n_groups * n_branches * n_channels_per_branch
        n_channels_per_group_out = n_channels_out / n_groups

        assert n_channels_out % n_groups == 0

        # slice maps into groups
        tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)])(tensor_input)

        t_outputs = []
        for idx_group in range(n_groups):
            tensor_group = tensors[idx_group]

            # branch 1: dimension reduction only and no temporal conv
            t_0 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_0 = BatchNormalization()(t_0)

            # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
            t_3 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_3 = DepthwiseConv1DLayer(3, padding='same')(t_3)
            t_3 = BatchNormalization()(t_3)

            # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
            t_5 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_5 = DepthwiseConv1DLayer(5, padding='same')(t_5)
            t_5 = BatchNormalization()(t_5)

            # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
            t_7 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_7 = DepthwiseConv1DLayer(7, padding='same')(t_7)
            t_7 = BatchNormalization()(t_7)

            # branch 5: dimension reduction followed by temporal max pooling
            t_1 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_1 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same')(t_1)
            t_1 = BatchNormalization()(t_1)

            # concatenate channels of branches
            tensor = Concatenate(axis=4)([t_0, t_3, t_5, t_7, t_1])
            t_outputs.append(tensor)

        # concatenate channels of groups
        tensor = Concatenate(axis=4)(t_outputs)
        tensor = Activation('relu')(tensor)

        # shuffle channels
        tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group_out))(tensor)
        tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
        tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels_out))(tensor)

        return tensor

    def __inception_style_temporal_layer_shuffled_depthwise_dilated_complicated(self, tensor_input, n_channels_per_branch):
        n_groups = self.n_groups
        _, n_timesteps, side_dim1, side_dim2, n_channels_in = K.int_shape(tensor_input)
        assert n_channels_in % n_groups == 0
        n_branches = 5

        n_channels_per_group_in = n_channels_in / n_groups
        n_channels_out = n_groups * n_branches * n_channels_per_branch
        n_channels_per_group_out = n_channels_out / n_groups

        assert n_channels_out % n_groups == 0

        # slice maps into groups
        tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)])(tensor_input)

        t_outputs = []
        for idx_group in range(n_groups):
            tensor_group = tensors[idx_group]

            # branch 1: dimension reduction only and no temporal conv
            t_0 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_0 = BatchNormalization()(t_0)

            # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 1)
            t_3 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_3 = DepthwiseDilatedConv1DLayer(3, 1, padding='same')(t_3)
            t_3 = BatchNormalization()(t_3)

            # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 2)
            t_5 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_5 = DepthwiseDilatedConv1DLayer(3, 2, padding='same')(t_5)
            t_5 = BatchNormalization()(t_5)

            # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 3, dilation 3)
            t_7 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_7 = DepthwiseDilatedConv1DLayer(3, 3, padding='same')(t_7)
            t_7 = BatchNormalization()(t_7)

            # # branch 5: dimension reduction followed by temporal max pooling
            t_1 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same')(tensor_group)
            t_1 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same')(t_1)
            t_1 = BatchNormalization()(t_1)

            # concatenate channels of branches
            tensor = Concatenate(axis=4)([t_0, t_3, t_5, t_7, t_1])
            t_outputs.append(tensor)

        # concatenate channels of groups
        tensor = Concatenate(axis=4)(t_outputs)
        tensor = Activation('relu')(tensor)

        # shuffle channels
        tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group_out))(tensor)
        tensor = TransposeLayer((0, 1, 2, 3, 5, 4))(tensor)
        tensor = ReshapeLayer((n_timesteps, side_dim1, side_dim2, n_channels_out))(tensor)

        return tensor

    def __global_spatial_pooling(self, tensor):
        tensor_shape = tensor.get_shape().as_list()
        assert len(tensor_shape) == 5

        # avg pool and max pool, and concat them to spatial dimension
        tensor_ap = MaxLayer(axis=2, is_keep_dim=True)(tensor)
        tensor_mp = MeanLayer(axis=2, is_keep_dim=True)(tensor)
        tensor = Concatenate(axis=2)([tensor, tensor_ap, tensor_mp])
        return tensor

    def __get_n_channels_per_branch(self, n_channels_in):
        n_branches = 5
        n_groups = self.n_groups
        expansion_factor = self.expansion_factor

        n_channels_per_branch = n_channels_in * expansion_factor / float(n_branches * n_groups)
        n_channels_per_branch = int(n_channels_per_branch)
        n_channels_out = n_channels_per_branch * n_groups * n_branches
        n_channels_out = int(n_channels_out)
        return n_channels_per_branch, n_channels_out

# endregion