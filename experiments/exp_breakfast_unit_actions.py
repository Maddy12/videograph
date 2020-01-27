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

"""
This experiment is for Breakfast datasetm multi-label classification of unit-actions.
"""

import time
import datetime

import numpy as np
from sklearn.metrics import average_precision_score

import tensorflow as tf
from keras.layers import Input, BatchNormalization, Dense, LeakyReLU, Dropout, Conv3D, Activation
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import multi_gpu_utils
from keras.callbacks import Callback

from nets.keras_layers import MaxLayer, MeanLayer

from nets import videograph
from core import utils, keras_utils
from core.utils import Path as Pth
from core import const as c

# region Constants

N_CLASSES = 48

# endregion

# region Train

def train_model_on_pickled_features():
    """
    Train model.
    """

    model_type = 'i3d_rgb'
    feature_type = 'mixed_5c'

    n_centroids = 128
    n_timesteps = 64
    is_spatial_pooling = True
    is_resume_training = False

    batch_size_tr = 12
    batch_size_te = 30
    n_epochs = 100
    n_classes = N_CLASSES
    n_gpus = 1

    model_name = 'classifier_%s' % (utils.timestamp())
    model_weight_path = ''
    features_path = Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames_max_pool.h5', (n_timesteps * 8,)) if is_spatial_pooling else Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames.h5', (n_timesteps * 8,))
    centroids_path = Pth('Breakfast/features_centroids/features_random_%d_centroids.pkl', (n_centroids,))
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions.pkl')
    (_, y_tr), (_, y_te) = utils.pkl_load(gt_actions_path)
    centroids = utils.pkl_load(centroids_path)

    n_feat_maps, feat_map_side_dim = utils.get_model_feat_maps_info(model_type, feature_type)
    feat_map_side_dim = 1 if is_spatial_pooling else feat_map_side_dim
    input_shape = (None, n_timesteps, feat_map_side_dim, feat_map_side_dim, n_feat_maps)

    print ('--- start time')
    print (datetime.datetime.now())

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    root_model, model = __load_model_mlp_classifier_video_graph(centroids, n_classes, input_shape, n_gpus=n_gpus, is_load_weights=is_resume_training, weight_path=model_weight_path)
    t2 = time.time()
    duration = t2 - t1
    print (root_model.summary(line_length=130, positions=None, print_fn=None))
    print ('... model built, duration (sec): %d' % (duration))

    # load data
    print ('... loading data: %s' % (features_path))
    print ('... centroids: %s' % (centroids_path))
    t1 = time.time()
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    t2 = time.time()
    duration = t2 - t1
    print ('... data loaded: %d' % (duration))
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_te.shape)
    print(y_te.shape)

    n_tr = len(x_tr)
    n_te = len(x_te)
    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print ('... [tr]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr, n_gpus))
    print ('... [te]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_te, n_batch_te, batch_size_te, n_gpus))

    score_callback = ScoreCallback(model, None, None, x_te, y_te, batch_size_te)
    callbacks = [score_callback]
    model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size_tr, validation_split=0.0, validation_data=(x_te, y_te), shuffle=True, callbacks=callbacks, verbose=2)
    print ('--- finish time')
    print (datetime.datetime.now())

# endregion

# region Models

def __load_model_mlp_classifier_video_graph(centroids, n_classes, input_shape_x, n_gpus, is_load_weights, weight_path):
    """
    Model
    """

    # optimizer and loss
    loss = keras_utils.LOSSES[3]
    output_activation = keras_utils.ACTIVATIONS[2]
    optimizer = SGD(lr=0.01)
    optimizer = Adam(lr=0.01, epsilon=1e-8)
    optimizer = Adam(lr=0.01, epsilon=1e-4)

    expansion_factor = 5.0 / 4.0
    n_groups = int(input_shape_x[-1] / 128.0)

    # per-layer kernel size and max pooling for centroids and timesteps
    n_graph_layers = 2

    # time kernel
    t_kernel_size = 7
    t_max_size = 3

    # node kernel
    c_kernel_size = 7
    c_max_size = 3
    c_avg_size = 4

    # space kernel
    s_kernel_size = 2
    s_kernel_size = 1

    n_centroids, _ = centroids.shape

    _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape_x
    t_input_x = Input(shape=(n_timesteps, side_dim, side_dim, n_channels_in), name='input_x')  # (None, 64, 1024)
    t_input_c = Input(tensor=tf.constant(centroids, dtype=tf.float32), name='input_n')  # (1, 100, 1024)
    tensor = t_input_x

    # spatial convolution
    n_channels_in = 1024
    tensor = Conv3D(n_channels_in, (1, s_kernel_size, s_kernel_size), padding='VALID', name='conv_s')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)

    # pool over space
    tensor = MaxLayer(axis=(2, 3), is_keep_dim=True, name='global_pool_s')(tensor)  # (None, 64, 7, 7, 1024)

    # centroid-attention
    tensor = videograph.node_attention(tensor, t_input_c, n_channels_in, activation_type='relu')  # (N, 100, 64, 7, 7, 1024)

    # graph embeddings
    tensor = videograph.graph_embedding(tensor, n_graph_layers, c_avg_size, c_kernel_size, t_kernel_size, c_max_size, t_max_size)  # (N, 100, 64, 7, 7, 1024)

    # centroid pooling
    tensor = MeanLayer(axis=(1,), name='global_pool_n')(tensor)

    # temporal pooling
    tensor = MaxLayer(axis=(1, 2, 3), name='global_pool_t')(tensor)

    # activity classification
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(512)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(n_classes)(tensor)
    t_output = Activation(output_activation)(tensor)

    model = Model(input=[t_input_x, t_input_c], output=t_output)

    if is_load_weights:
        model.load_weights(weight_path)

    if n_gpus == 1:
        model.compile(loss=loss, optimizer=optimizer)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer)

    return model, parallel_model

# endregion

# region Callbacks

class ScoreCallback(Callback):
    def __init__(self, model, x_tr, y_tr, x_te, y_te, batch_size):

        self.model = model
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):

        y_pred_te = self.model.predict(self.x_te)
        if self.x_tr is not None and self.y_tr is not None:
            y_pred_tr = self.model.predict(self.x_tr, self.batch_size)
            a_tr = self.__mean_avg_precision(self.y_tr, y_pred_tr)
            a_te = self.__mean_avg_precision(self.y_te, y_pred_te)
            a_tr *= 100.0
            a_te *= 100.0
            msg = '        map_: %.02f%%, %.02f%%' % (a_tr, a_te)
            print (msg)
        else:
            a_te = self.__mean_avg_precision(self.y_te, y_pred_te)
            a_te *= 100.0
            msg = '        map_: %.02f%%' % (a_te)
            print (msg)

    def __mean_avg_precision(self, y_true, y_pred):
        # """ Returns mAP """
        map = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(N_CLASSES)]
        map = np.nan_to_num(map)
        map = np.mean(map)
        return map

# endregion
