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
This experiment is for Breakfast dataset, single-label classifications of activities.
"""

import random
import os
import sys
import time
import datetime
import threading
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score
from MulticoreTSNE import MulticoreTSNE

import tensorflow as tf
import keras.backend as K
from keras import callbacks
from keras.layers import Input, BatchNormalization, Dense, LeakyReLU, Conv3D, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import multi_gpu_utils
from keras.callbacks import Callback

from nets.keras_layers import MaxLayer, MeanLayer, ReshapeLayer, NetVLAD
from nets import videograph, timeception

from datasets import ds_breakfast
from core import utils, keras_utils, configs, data_utils
from core import const as c
from core.utils import Path as Pth

# region Const

N_CLASSES = 10

# endregion

# region Train

def train_model_on_pickled_features():
    """
    Train model.
    """

    model_type = 'i3d_rgb'
    feature_type = 'mixed_5c'
    is_spatial_pooling = False
    is_resume_training = False

    n_timesteps = 64
    batch_size_tr = 16
    batch_size_te = 40
    n_centroids = 128
    n_epochs = 100
    n_classes = N_CLASSES
    n_gpus = 1

    model_name = 'classifier_%s' % (utils.timestamp())
    model_weight_path = ''
    model_root_path = Pth('Breakfast/models/')
    gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
    centroids_path = Pth('Breakfast/features_centroids/features_random_%d_centroids.pkl', (n_centroids,))
    features_path = Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames_max_pool.h5', (n_timesteps * 8,)) if is_spatial_pooling else Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames.h5', (n_timesteps * 8,))

    centroids = utils.pkl_load(centroids_path)
    (video_ids_tr, y_tr), (video_ids_te, y_te) = utils.pkl_load(gt_activities_path)

    n_feat_maps, feat_map_side_dim = __get_model_feat_maps_info(model_type, feature_type)
    feat_map_side_dim = 1 if is_spatial_pooling else feat_map_side_dim
    input_shape = (None, n_timesteps, feat_map_side_dim, feat_map_side_dim, n_feat_maps)

    print ('--- start time')
    print (datetime.datetime.now())

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()

    # root_model, model = __load_model_mlp_classifier_action_vlad(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=is_resume_training, weight_path=model_weight_path)
    # root_model, model = __load_model_mlp_classifier_timeception(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=is_resume_training, weight_path=model_weight_path)
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

    n_tr = len(x_tr)
    n_te = len(x_te)
    n_batch_tr = __calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = __calc_num_batches(n_te, batch_size_te)
    print ('... [tr]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr, n_gpus))
    print ('... [te]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_te, n_batch_te, batch_size_te, n_gpus))

    save_callback = keras_utils.ModelSaveCallback(model, model_name, model_root_path)
    model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size_tr, validation_split=0.0, validation_data=(x_te, y_te), shuffle=True, callbacks=[save_callback], verbose=2)
    print ('--- finish time')
    print (datetime.datetime.now())

def train_model_on_video_features_i3d():
    """
    Train model of features stored on local disc.
    """

    model_type = 'i3d_rgb'
    feature_type = 'mixed_5c'
    is_spatial_pooling = False
    is_spatial_max = False
    is_save = True
    n_gpus = 1

    batch_size_tr = 20
    batch_size_te = 30
    n_threads = 20
    n_epochs = 500
    n_classes = N_CLASSES
    n_centroids = 128
    n_timesteps = 64
    n_frames = n_timesteps * 8

    model_name = 'classifier_%s' % (utils.timestamp())
    model_weight_path = ''

    # resnet-152
    features_root_path = Pth('Breakfast/features_i3d_mixed_5c_%s_frames', (n_frames))
    centroids_path = Pth('Breakfast/features_centroids/features_random_%d_centroids.pkl', (n_centroids,))
    video_annot_path = Pth('Breakfast/annotation/gt_activities.pkl')
    centroids = utils.pkl_load(centroids_path)

    n_feat_maps, feat_map_side_dim = __get_model_feat_maps_info(model_type, feature_type)
    feat_map_side_dim = 1 if is_spatial_pooling else feat_map_side_dim
    input_shape = (None, n_timesteps, feat_map_side_dim, feat_map_side_dim, n_feat_maps)

    print ('--- start time')
    print (datetime.datetime.now())

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()

    # root_model, model = __load_model_mlp_classifier_conv_pool(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path=model_weight_path)
    # root_model, model = __load_model_mlp_classifier_action_vlad(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path=model_weight_path)
    # root_model, model = __load_model_mlp_classifier_timeception(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path=model_weight_path)
    root_model, model = __load_model_mlp_classifier_video_graph(centroids, n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path=model_weight_path)

    t2 = time.time()
    duration = t2 - t1
    print (root_model.summary(line_length=130, positions=None, print_fn=None))
    print ('... model built, duration (sec): %d' % (duration))

    # load data
    print ('... loading data: %s' % (features_root_path))
    t1 = time.time()
    (v_names_tr, y_tr), (v_names_vl, y_vl), (v_names_te, y_te) = utils.pkl_load(video_annot_path)
    v_names_tr = np.hstack((v_names_tr, v_names_vl))
    y_tr = np.hstack((y_tr, y_vl))
    del v_names_vl
    del y_vl
    action_ids = np.arange(1, N_CLASSES + 1)
    y_tr = utils.label_binarize(y_tr, action_ids)
    y_te = utils.label_binarize(y_te, action_ids)
    n_tr = len(v_names_tr)
    n_te = len(v_names_te)
    n_batch_tr = keras_utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = keras_utils.calc_num_batches(n_te, batch_size_te)
    t2 = time.time()
    print ('... centroids: %s' % (centroids_path))
    print ('... data loaded: %d' % (t2 - t1))

    print ('... [tr]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr, n_gpus))
    print ('... [te]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_te, n_batch_te, batch_size_te, n_gpus))

    # load features async
    async_loader_tr = data_utils.AsyncVideoFeaturesLoaderBreakfast(features_root_path, y_tr, n_timesteps, batch_size_tr, n_feat_maps, feat_map_side_dim, n_threads)
    async_loader_te = data_utils.AsyncVideoFeaturesLoaderBreakfast(features_root_path, y_te, n_timesteps, batch_size_te, n_feat_maps, feat_map_side_dim, n_threads)

    # shuffle the data for the first time
    async_loader_tr.shuffle_data()

    # start getting images ready for the first barch
    async_loader_tr.load_feats_in_batch(1)
    async_loader_te.load_feats_in_batch(1)

    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1

        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        tt1 = time.time()
        waiting_duration_total = 0

        # loop and train
        for idx_batch in range(n_batch_tr):

            batch_num = idx_batch + 1

            # wait untill the image_batch is loaded
            t1 = time.time()
            while async_loader_tr.is_busy():
                threading._sleep(0.1)
            t2 = time.time()

            # get batch of training samples
            x_tr_b, y_tr_b = async_loader_tr.get_batch_data()

            # start getting the next image_batch ready
            if batch_num < n_batch_tr:
                next_batch_num = batch_num + 1
                async_loader_tr.load_feats_in_batch(next_batch_num)

            # train and get predictions
            loss_batch_tr, acc_batch_tr = model.train_on_batch(x_tr_b, y_tr_b)

            loss_tr += loss_batch_tr
            acc_tr += acc_batch_tr
            loss_tr_b = loss_batch_tr / float(batch_num)
            acc_tr_b = 100 * acc_batch_tr / float(batch_num)

            tt2 = time.time()
            duration = tt2 - tt1
            waiting_duration = t2 - t1
            waiting_duration_total += waiting_duration
            sys.stdout.write('\r%04ds - epoch: %02d/%02d, batch [tr]: %02d/%02d, loss_tr: %.02f, acc_tr: %.02f, waited: %.01f       ' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_tr_b, acc_tr_b, waiting_duration))

        # loop and test
        for idx_batch in range(n_batch_te):

            batch_num = idx_batch + 1

            # wait untill the image_batch is loaded
            t1 = time.time()
            while async_loader_te.is_busy():
                threading._sleep(0.1)
            t2 = time.time()

            # get batch of testing samples
            x_te_b, y_te_b = async_loader_te.get_batch_data()

            # start getting the next image_batch ready
            if batch_num < n_batch_te:
                next_batch_num = batch_num + 1
                async_loader_te.load_feats_in_batch(next_batch_num)

            # test and get predictions
            loss_batch_te, acc_batch_te = model.test_on_batch(x_te_b, y_te_b)

            loss_te += loss_batch_te
            acc_te += acc_batch_te
            loss_te_b = loss_batch_te / float(batch_num)
            acc_te_b = 100 * acc_batch_te / float(batch_num)

            tt2 = time.time()
            duration = tt2 - tt1
            waiting_duration = t2 - t1
            waiting_duration_total += waiting_duration
            sys.stdout.write('\r%04ds - epoch: %02d/%02d, batch [te]: %02d/%02d, loss_te: %.02f, acc_te: %.02f, waited: %.01f  ' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_te_b, acc_te_b, waiting_duration))

        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)

        acc_tr /= float(n_batch_tr)
        acc_te /= float(n_batch_te)

        acc_tr *= 100.0
        acc_te *= 100.0

        tt2 = time.time()
        duration = tt2 - tt1
        sys.stdout.write('\r%04ds - epoch: %02d/%02d, loss_tr %.02f, acc_tr %.02f, loss_te %.02f, acc_te: %.02f, waited: %d   \n' % (duration, epoch_num, n_epochs, loss_tr, acc_tr, loss_te, acc_te, waiting_duration_total))

        # shuffle the data
        async_loader_tr.shuffle_data()

        # because we setted a new data list, start getting the first batch
        async_loader_tr.load_feats_in_batch(1)
        async_loader_te.load_feats_in_batch(1)

        # save the model, if required
        if is_save:
            __save_model(root_model, model_name, epoch_num)

    print ('--- finish time')
    print (datetime.datetime.now())

# endregion

# region Models

def __load_model_mlp_classifier_action_vlad(n_classes, input_shape, n_gpus, is_load_weights, weight_path):
    """
    Model
    """

    # optimizer and loss
    loss = keras_utils.LOSSES[0]
    metrics = [keras_utils.METRICS[0]]
    output_activation = keras_utils.ACTIVATIONS[3]
    optimizer = SGD(lr=0.01)
    optimizer = Adam(lr=0.01, epsilon=1e-8)
    optimizer = Adam(lr=0.01, epsilon=1e-4)

    expansion_factor = 5.0 / 4.0
    _, n_timesteps, side_dim, _, n_channels_in = input_shape

    input_shape = (input_shape[1:])
    t_input = Input(shape=input_shape)  # (None, 7, 7, 1024)
    tensor = t_input

    # spatial convolution
    n_channels_out = 512
    tensor = Conv3D(n_channels_out, kernel_size=(1, 1, 1), padding='same')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)
    n_channels_in = n_channels_out

    # reshape for vlad
    tensor = ReshapeLayer((n_channels_in,))(tensor)

    # vlad layer
    max_samples = n_timesteps * side_dim * side_dim
    tensor = NetVLAD(n_channels_in, max_samples, 32)(tensor)

    # dense layers
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(256)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(n_classes)(tensor)

    t_output = Activation(output_activation)(tensor)
    model = Model(input=t_input, output=t_output)

    if is_load_weights:
        model.load_weights(weight_path)

    if n_gpus == 1:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model, parallel_model

def __load_model_mlp_classifier_timeception(n_classes, input_shape, n_gpus, is_load_weights, weight_path):
    """
    Model
    """
    # optimizer and loss
    loss = keras_utils.LOSSES[0]
    metrics = [keras_utils.METRICS[0]]
    output_activation = keras_utils.ACTIVATIONS[3]
    optimizer = SGD(lr=0.01)
    optimizer = Adam(lr=0.01, epsilon=1e-8)
    optimizer = Adam(lr=0.01, epsilon=1e-4)

    n_tc_layer = 3
    expansion_factor = 5.0 / 4.0
    _, n_timesteps, side_dim, _, n_channels_in = input_shape
    n_groups = int(n_channels_in / 128.0)
    print ('... n_groups, expansion factor: %d, %.02f' % (n_groups, expansion_factor))

    input_shape = (input_shape[1:])
    t_input = Input(shape=input_shape)  # (None, 20, 7, 7, 1024)
    tensor = t_input

    # timeception layers
    tensor = timeception.timeception_temporal_convolutions(tensor, n_tc_layer, n_groups, expansion_factor, is_dilated=True)

    # spatio-temporal pooling
    tensor = MaxLayer(axis=(1, 2, 3))(tensor)

    # dense layers
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(512)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(n_classes)(tensor)
    t_output = Activation(output_activation)(tensor)
    model = Model(input=t_input, output=t_output)

    if is_load_weights:
        model.load_weights(weight_path)

    if n_gpus == 1:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model, parallel_model

def __load_model_mlp_classifier_video_graph(centroids, n_classes, input_shape_x, n_gpus, is_load_weights, weight_path):
    """
    Model
    """

    # optimizer and loss
    loss = keras_utils.LOSSES[0]
    metrics = [keras_utils.METRICS[0]]
    output_activation = keras_utils.ACTIVATIONS[3]
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
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model, parallel_model

# endregion

# region Misc

def __shuffle_data(x, y):
    idx = range(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y

def __get_model_feat_maps_info(model_type, feature_type):
    if model_type in ['vgg']:
        if feature_type in ['pool5']:
            return 512, 7
        elif feature_type in ['conv5_3']:
            return 512, 14
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['resnet_152', 'resnet3d']:
        if feature_type in ['res4b35']:
            return 1024, 14
        elif feature_type in ['res5c', 'res52']:
            return 2048, 7
        elif feature_type in ['pool5']:
            return 2048, 1
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['i3d_rgb']:
        if feature_type in ['mixed_5c']:
            return 1024, 7
        elif feature_type in ['softmax']:
            return 400, 1
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['i3d_kinetics_keras']:
        if feature_type in ['mixed_4f']:
            return 832, 7
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    else:
        raise Exception('Sorry, unsupported model type: %s' % (model_type))

def __calc_num_batches(n_samples, batch_size):
    n_batch = int(n_samples / float(batch_size))
    n_batch = n_batch if n_samples % batch_size == 0 else n_batch + 1
    return n_batch

def __save_model(root_model, model_name, epoch_num):
    root_path = c.DATA_ROOT_PATH
    model_root_path = '%s/Charades/models/%s' % (root_path, model_name)
    if not os.path.exists(model_root_path):
        os.mkdir(model_root_path)

    model_path = '%s/%03d.model' % (model_root_path, epoch_num)
    model_json_path = '%s/%03d.json' % (model_root_path, epoch_num)
    model_weight_path = '%s/%03d.pkl' % (model_root_path, epoch_num)

    # for very long model, this does not work
    # self.root_model.save(model_path)
    # only save model definition and weights
    keras_utils.save_model(root_model, model_json_path, model_weight_path)

def __shuffle_temporal_order(x):
    x_shape = x.shape
    assert len(x_shape) == 5
    n_timesteps = x_shape[1]

    idx = np.random.randint(0, n_timesteps, (n_timesteps,))
    x = x[:, idx]
    return x

# endregion
