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
This experiment is for EPIC-Kitchens dataset.
"""

import sys
import os
import random
import time
import datetime
import threading
import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LeakyReLU, Dropout, Activation, Conv3D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import multi_gpu_utils

from nets.keras_layers import ReshapeLayer, TransposeLayer, DepthwiseDilatedConv1DLayer, DepthwiseConv1DLayer, MaxLayer, MeanLayer, NetVLAD
from nets.keras_layers import DepthwiseDenseLayer, ConvOverSpaceLayer

from nets.i3d_keras_epic_kitchens import Inception_Inflated3d_Backbone
from datasets import ds_epic_kitchens
from nets import videograph, timeception
from core import utils, keras_utils, metrics, image_utils
from core.utils import Path as Pth

# region Train

def train_model_on_pickled_features():
    """
    Train model.
    """

    annotation_type = 'noun'
    annot_path = Pth('EPIC-Kitchens/annotation/annot_video_level_many_shots.pkl')
    (y_tr, y_te), n_classes = __load_annotation(annot_path, annotation_type)

    model_type = 'i3d_rgb'
    feature_type = 'mixed_5c'
    n_nodes = 128
    n_timesteps = 64
    n_frames_per_segment = 8
    n_frames_per_video = n_timesteps * n_frames_per_segment
    batch_size_tr = 20
    batch_size_te = 30
    n_epochs = 500
    epoch_offset = 0
    model_name = 'classifier_%s' % (utils.timestamp())
    model_root_path = Pth('EPIC-Kitchens/models')

    features_path = Pth('EPIC-Kitchens/features/features_i3d_mixed_5c_%d_frames.h5', (n_frames_per_video,))
    nodes_path = Pth('EPIC-Kitchens/features_centroids/features_random_%d.pkl', (n_nodes,))
    n_channels, side_dim = utils.get_model_feat_maps_info(model_type, feature_type)
    input_shape = (None, n_timesteps, side_dim, side_dim, n_channels)
    nodes = utils.pkl_load(nodes_path)

    print ('--- start time')
    print (datetime.datetime.now())

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = __load_model_videograph(nodes, n_classes, input_shape)
    t2 = time.time()
    duration = t2 - t1
    print (model.summary(line_length=130, positions=None, print_fn=None))
    print ('... model built, duration (sec): %d' % (duration))

    # load data
    print ('... loading data: %s' % (features_path))
    t1 = time.time()
    # features are extracting using datasets.Epic_Kitchens.i3d_keras_epic_kitchens()
    # we use out-of-box i3d (pre-trained on kinetics, NOT fine-tuned on epic-kitchens) with last conv feature 7*7*1024 'mixed_5c'
    # to get a better performance, you need to write code to randomly sample new frames and extract their features every new epoch
    # please use this function to random sampling, instead of uniform sampling: Epic_Kitchens.__random_sample_frames_per_video_for_i3d()
    # then extract their features, as done in: Epic_Kitchens._901_extract_features_i3d()
    # then train on the extracted features. Please do so in every epoch. It's computationally heavy, but you cannot avoid random sampling to get better results.
    # Even better results if you replace I3D with a 2D/3D CNN that's previously fine-tuned on Epic-Kitchens
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    t2 = time.time()

    duration = t2 - t1
    print ('... data loaded: %d' % (duration))

    n_tr = len(x_tr)
    n_te = len(x_te)
    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print ('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print ('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))
    print (x_tr.shape)
    print (x_te.shape)
    print (y_tr.shape)
    print (y_te.shape)

    save_callback = keras_utils.ModelSaveCallback(model, model_name, epoch_offset, model_root_path)
    score_callback = keras_utils.MapScoreCallback(model, None, None, x_te, y_te, batch_size_te, n_classes)
    model_callbacks = [save_callback, score_callback]
    model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size_tr, validation_split=0.0, validation_data=(x_te, y_te), shuffle=True, callbacks=model_callbacks, verbose=2)

    print ('--- finish time')
    print (datetime.datetime.now())

def train_model_on_video_frames():
    """
    When training model on images, the model won't fit in gpu.
    If trained on several gpus, the batch size will get so small that BatchNorm is not applicable anymore.
    The solution is to use first 3 gpus to extract features from the backbone model (i.e. bottom part, for example: I3D or ResNet),
    and to use the 4th gpu to train our model (i.e. top part) on these features.
    """

    # this is to allow for small cpu utilization by numpy
    # has to be set before importing numpy
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"

    # if training from scratch
    resume_epoch_num = 0
    is_resume_training = False
    resume_timestamp = ''

    # get the model part to run
    timestamp = utils.timestamp() if not is_resume_training else resume_timestamp
    starting_epoch_num = 0 if not is_resume_training else resume_epoch_num
    n_epochs = 500

    # for i3d-keras
    n_centroids = 128
    n_frames_bottom = 512
    n_frames_top = 64
    n_instances = 3
    model_bottom = __start_train_model_on_video_frames_backbone_i3d_keras
    model_top = __start_train_model_on_video_frames_videograph

    # also, create the files where the training state will be stored
    global TRAIN_STATE
    TRAIN_STATE = TrainingState()

    # bottom part, instance 1
    args_bottom_1 = (n_epochs, starting_epoch_num, n_frames_bottom, n_instances, 1)
    thread_bottom_1 = threading.Thread(target=model_bottom, args=args_bottom_1)

    # bottom part, instance 2
    args_bottom_2 = (n_epochs, starting_epoch_num, n_frames_bottom, n_instances, 2)
    thread_bottom_2 = threading.Thread(target=model_bottom, args=args_bottom_2)

    # bottom part, instance 3
    args_bottom_3 = (n_epochs, starting_epoch_num, n_frames_bottom, n_instances, 3)
    thread_bottom_3 = threading.Thread(target=model_bottom, args=args_bottom_3)

    # top part
    args_top = (n_epochs, n_frames_top, n_centroids, timestamp, is_resume_training, starting_epoch_num)
    thread_top = threading.Thread(target=model_top, args=args_top)

    thread_top.start()
    thread_bottom_1.start()
    thread_bottom_2.start()
    thread_bottom_3.start()

    thread_top.join()
    thread_bottom_1.join()
    thread_bottom_2.join()
    thread_bottom_3.join()

def __start_train_model_on_video_frames_videograph(n_epochs, n_timesteps, n_centroids, timestamp, is_resume_training, start_epoch_num):
    # configure the gpu to be used by keras
    gpu_core_id = 3
    device_id = '/gpu:%d' % gpu_core_id

    # with graph.as_default():
    # with session.as_default():

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config, graph=graph)
    K.set_session(sess)
    with sess:
        with tf.device(device_id):
            __train_model_on_video_frames_videograph(n_epochs, n_timesteps, n_centroids, timestamp, is_resume_training, start_epoch_num)

def __start_train_model_on_video_frames_backbone_i3d_keras(n_epochs, starting_epoch_num, n_frames_per_video, n_instances, instance_num):
    # configure the gpu to be used by keras
    gpu_core_id = instance_num - 1
    device_id = '/gpu:%d' % gpu_core_id

    assert instance_num in [1, 2, 3], 'Sorry, wrong instance number: %d' % (instance_num)

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config, graph=graph)
    K.set_session(sess)
    with sess:
        with tf.device(device_id):
            __train_model_on_video_frames_backbone_i3d_keras(n_epochs, starting_epoch_num, n_frames_per_video, n_instances, instance_num)

def __train_model_on_video_frames_videograph(n_epochs, n_timesteps, n_centroids, timestamp, is_resume_training, start_epoch_num):
    """
    Train model of 3rd gpu, train it on features extracted on first 2 gpus.
    """

    global TRAIN_STATE
    assert (start_epoch_num > 1 and is_resume_training) or (start_epoch_num == 0 and not is_resume_training), 'sorry, either provide resume_epoch_num or set the model as not resuming with resume_epoch_num = 0'

    n_frames_per_segment = 8
    n_frames_per_video = n_frames_per_segment * n_timesteps

    # locations
    model_name = 'classifier_from_video_frames_%s' % (timestamp)
    resume_model_json_path = Pth('EPIC-Kitchens/models/%s/%03d.json', (model_name, start_epoch_num))
    resume_model_weights_path = Pth('EPIC-Kitchens/models/%s/%03d.pkl', (model_name, start_epoch_num))

    frames_root_path = Pth('EPIC-Kitchens/frames_rgb_resized/train')
    features_te_path = Pth('EPIC-Kitchens/features/features_i3d_mixed_5c_%d_frames_te.h5', (n_frames_per_video,))
    centroids_path = Pth('EPIC-Kitchens/features_centroid/features_random_%d_centroids.pkl', (n_centroids,))
    centroids_path = Pth('EPIC-Kitchens/features_centroid/features_sobol_%d_centroids.pkl', (n_centroids,))
    video_names_splits_path = Pth('EPIC-Kitchens/annotation/video_names_splits.pkl')
    frame_relative_pathes_dict_path = Pth('EPIC-Kitchens/annotation/frame_relative_pathes_dict_tr.pkl')
    annot_path = Pth('EPIC-Kitchens/annotation/annot_video_level_many_shots.pkl')

    is_save_centroids = False
    is_save_model = True
    verbose = False

    n_gpus = 1
    n_classes = ds_epic_kitchens.N_NOUNS_MANY_SHOT

    batch_size_tr = 20
    batch_size_te = 40
    n_threads_te = 16

    n_feat_maps = 1024
    featmap_side_dim = 7
    input_shape = (None, n_timesteps, featmap_side_dim, featmap_side_dim, n_feat_maps)

    # load centroids
    centroids = utils.pkl_load(centroids_path)

    print ('--- start time')
    print (datetime.datetime.now())

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()

    # load new or previous model
    if is_resume_training:
        custom_objects = {'DepthwiseDilatedConv1DLayer': DepthwiseDilatedConv1DLayer,
                          'DepthwiseConv1DLayer': DepthwiseConv1DLayer,
                          'DepthwiseDenseLayer': DepthwiseDenseLayer,
                          'ConvOverSpaceLayer': ConvOverSpaceLayer,
                          'TransposeLayer': TransposeLayer,
                          'ReshapeLayer': ReshapeLayer,
                          'MeanLayer': MeanLayer,
                          'MaxLayer': MaxLayer}
        model = keras_utils.load_model(resume_model_json_path, resume_model_weights_path, custom_objects=custom_objects, is_compile=False)
        model, _ = __compile_model_for_finetuning(model, n_gpus)
    else:
        model, _ = __load_model_action_vlad(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path='')
        model, _ = __load_model_videograph(centroids, n_classes, input_shape)
        # model, _ = __load_model_timeception(n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path='')
        # model, _ = __load_model_mlp_classifier_transformer_centroids_with_graph_embedding(centroids, n_classes, input_shape, n_gpus=n_gpus, is_load_weights=False, weight_path='')

    # dry run to get the model loaded in gpu
    dummy_feature = np.zeros(tuple([batch_size_tr] + list(input_shape[1:])), dtype=np.float32)
    model.predict(dummy_feature)

    t2 = time.time()
    duration = t2 - t1
    print (model.summary(line_length=120, positions=None, print_fn=None))
    print ('... model built, duration (sec): %d' % (duration))

    # load data
    print ('... loading data')
    t1 = time.time()

    (y_tr, _, _, y_te, _, _) = utils.pkl_load(annot_path)
    (video_names_tr, video_names_te) = utils.pkl_load(video_names_splits_path)
    frame_relative_pathes_dict = utils.pkl_load(frame_relative_pathes_dict_path)
    x_te = utils.h5_load(features_te_path)
    print ('... centroids: %s' % (centroids_path))

    n_tr = len(video_names_tr)
    n_te = len(video_names_te)

    # set list of video names and ground truth
    TRAIN_STATE.video_names_tr = video_names_tr
    TRAIN_STATE.class_nums_tr = y_tr

    # sample new frames
    sampled_video_frames_dict = ds_epic_kitchens.__random_sample_frames_per_video_for_i3d(TRAIN_STATE.video_names_tr, frames_root_path, frame_relative_pathes_dict, n_frames_per_segment, n_frames_per_video)
    TRAIN_STATE.video_frames_dict_tr = sampled_video_frames_dict

    del video_names_tr
    del video_names_te
    del y_tr

    n_batch_tr = keras_utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = keras_utils.calc_num_batches(n_te, batch_size_te)
    t2 = time.time()
    duration = t2 - t1
    print ('... data loaded: %d' % duration)
    print ('... [tr]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr, n_gpus))
    print ('... [te]: n, n_batch, batch_size, n_gpus: %d, %d, %d, %d' % (n_te, n_batch_te, batch_size_te, n_gpus))

    # make model top ready
    TRAIN_STATE.model_top_ready = True
    sys.stdout.write('\n')
    for idx_epoch in range(start_epoch_num, n_epochs):

        epoch_num = idx_epoch + 1
        # wait until bottom parts start
        while TRAIN_STATE.model_bottom_1_epoch_start < epoch_num or TRAIN_STATE.model_bottom_2_epoch_start < epoch_num or TRAIN_STATE.model_bottom_3_epoch_start < epoch_num:
            threading._sleep(2.0)
            if verbose:
                print ('... top part is waiting for bottom part to start extracting features for epoch %d' % (epoch_num))

        # epoch started, update counter
        TRAIN_STATE.model_top_epoch_start = epoch_num

        # video names are obtained from the state at the beginning of each epoch
        video_names_tr = TRAIN_STATE.video_names_tr
        y_tr = TRAIN_STATE.class_nums_tr

        loss_tr = 0.0
        loss_tr_b = 0.0
        tt1 = time.time()
        waiting_duration_total = 0

        # loop and train
        for idx_batch_tr in range(n_batch_tr):

            batch_num_tr = idx_batch_tr + 1

            start_idx_batch = idx_batch_tr * batch_size_tr
            stop_idx_batch = (idx_batch_tr + 1) * batch_size_tr
            video_names_tr_batch = video_names_tr[start_idx_batch:stop_idx_batch]
            y_tr_b = y_tr[start_idx_batch:stop_idx_batch]
            is_missing_features = True

            # wait until the festures are loaded
            t1 = time.time()
            while is_missing_features:
                is_missing_features = False
                for _v_name in video_names_tr_batch:
                    if _v_name not in TRAIN_STATE.feats_dict_tr_1 and _v_name not in TRAIN_STATE.feats_dict_tr_2 and _v_name not in TRAIN_STATE.feats_dict_tr_3:
                        is_missing_features = True
                        break
                if is_missing_features:
                    threading._sleep(1.0)
                    if verbose:
                        print ('... model top is waiting for missing videos: %s' % _v_name)
            t2 = time.time()

            x_tr_b = __get_features_from_dictionaries(video_names_tr_batch)
            x_tr_b = np.array(x_tr_b)

            loss_batch_tr = model.train_on_batch(x_tr_b, y_tr_b)

            # after training, remove feats from dictionary (# delete feature and remove key)
            for _v_name in video_names_tr_batch:
                if _v_name in TRAIN_STATE.feats_dict_tr_1:
                    del TRAIN_STATE.feats_dict_tr_1[_v_name]
                    TRAIN_STATE.feats_dict_tr_1.pop(_v_name, None)
                elif _v_name in TRAIN_STATE.feats_dict_tr_2:
                    del TRAIN_STATE.feats_dict_tr_2[_v_name]
                    TRAIN_STATE.feats_dict_tr_2.pop(_v_name, None)
                elif _v_name in TRAIN_STATE.feats_dict_tr_3:
                    del TRAIN_STATE.feats_dict_tr_3[_v_name]
                    TRAIN_STATE.feats_dict_tr_3.pop(_v_name, None)

            loss_tr += loss_batch_tr
            loss_tr_b = loss_tr / float(batch_num_tr)
            tt2 = time.time()
            duration = tt2 - tt1
            waiting_duration = t2 - t1
            waiting_duration_total += waiting_duration
            msg = '%04ds - epoch: %02d/%02d, batch [tr]: %02d/%02d, loss: %0.2f, waited: %.01f  ' % (duration, epoch_num, n_epochs, batch_num_tr, n_batch_tr, loss_tr_b, waiting_duration)
            if verbose:
                print(msg)
            else:
                sys.stdout.write('\r%s' % (msg))

        # test
        y_pred_te = model.predict(x_te, batch_size_te, verbose=0)
        map_te_avg = 100 * metrics.mean_avg_precision_sklearn(y_te, y_pred_te)
        loss_tr /= float(n_batch_tr)

        tt2 = time.time()
        duration = tt2 - tt1
        timestamp_now = utils.timestamp()
        msg = '%04ds - epoch: %02d/%02d, loss [tr]: %0.2f, map [te]: %0.2f%%, waited: %d, finished: %s   \n' % (duration, epoch_num, n_epochs, loss_tr, map_te_avg, waiting_duration_total, timestamp_now)
        if verbose:
            print(msg)
        else:
            sys.stdout.write('\r%s' % (msg))

        # after we're done with training and testing, shuffle the list of training videos, and set in the TRAINING_STATE, also sample new frames
        video_names_tr, y_tr = __shuffle_training_data(TRAIN_STATE.video_names_tr, TRAIN_STATE.class_nums_tr)
        TRAIN_STATE.video_names_tr = video_names_tr
        TRAIN_STATE.class_nums_tr = y_tr
        del video_names_tr, y_tr

        # also, sample new frames
        sampled_video_frames_dict = ds_epic_kitchens.__random_sample_frames_per_video_for_i3d(TRAIN_STATE.video_names_tr, frames_root_path, frame_relative_pathes_dict, n_frames_per_segment, n_frames_per_video)
        TRAIN_STATE.video_frames_dict_tr = sampled_video_frames_dict

        # update counter so the bottom part starts extracting features for the next epoch
        TRAIN_STATE.model_top_epoch_end = epoch_num

        # save the model and nodes, if required
        if is_save_model:
            __save_model(model, model_name, epoch_num)

        if is_save_centroids:
            __save_centroids(model, model_name, epoch_num)

    print ('--- finish time')
    print (datetime.datetime.now())

def __train_model_on_video_frames_backbone_i3d_keras(n_epochs, starting_epoch_num, n_frames_per_video, n_instances, instance_num):
    """
    Extract features from i3d-model to be used by our model.
    """

    verbose = False
    global TRAIN_STATE  # type: TrainingState
    assert instance_num in [1, 2, 3], 'Sorry, wrong instance number: %d' % (instance_num)
    assert n_instances == 3, 'Sorry, wrong number of instances %d' % (n_instances)

    n_threads = 16
    n_frames_per_segment = 8
    max_preloaded_feats = 40
    n_frames_in = n_frames_per_video
    n_frames_out = int(n_frames_in / float(n_frames_per_segment))
    assert n_frames_per_segment * n_frames_out == n_frames_in

    # load the model
    model = Inception_Inflated3d_Backbone()

    # reader for getting video frames
    video_reader = image_utils.AsyncImageReaderEpicKitchensForI3dKerasModel(n_threads=n_threads)

    # wait until model top is ready
    while not TRAIN_STATE.model_top_ready:
        threading._sleep(5.0)
        if verbose:
            print ('... bottom part (%d) is waiting for top part to get ready' % (instance_num))

    # extract features for n epoch
    for idx_epoch in range(starting_epoch_num, n_epochs):

        epoch_num = idx_epoch + 1

        video_frames_dict = TRAIN_STATE.video_frames_dict_tr
        video_names = TRAIN_STATE.video_names_tr
        n_videos = len(video_names)

        # only first instance can modify train_state and get videos from pickle
        if instance_num == 1:
            # model started, update count
            TRAIN_STATE.model_bottom_1_epoch_start = epoch_num
        elif instance_num == 2:
            # model started, update count
            TRAIN_STATE.model_bottom_2_epoch_start = epoch_num
        elif instance_num == 3:
            # model started, update count
            TRAIN_STATE.model_bottom_3_epoch_start = epoch_num
        else:
            raise Exception('Sorry, unknown instance number: %d' % (instance_num))

        if verbose:
            print ('epoch %d by instance %s' % (epoch_num, instance_num))

        # aync reader, and get load images for the first video, we will read the first group of videos
        current_video_name = video_names[instance_num - 1]
        current_video_frames = video_frames_dict[current_video_name]

        # just for clarification, can be reshaped from (256,) into (T, N) = (32, 8)
        # where T is the number of segments in one video, and N is the number of frames in one segment
        # video_group_frames = np.reshape(video_group_frames, tuple([n_frames_out, n_segment_length] + list(video_group_frames.shape[1:])))
        video_reader.load_imgs_in_batch(current_video_frames)

        # extract features only for training videos
        t1 = time.time()

        if verbose:
            print('... extracting features tr')
            print('... start time: %s' % utils.timestamp())

        # loop on list of videos
        for idx_video in range(n_videos):

            if instance_num == 1:
                # wait looping if there are so many features in the dictionary
                while len(TRAIN_STATE.feats_dict_tr_1) > max_preloaded_feats:
                    threading._sleep(1.0)
                    if verbose:
                        print ('... bottom part (%d) is waiting for features in the dictionary to get consumed by top part' % (instance_num))

            elif instance_num == 2:
                # wait looping if there are so many features in the dictionary
                while len(TRAIN_STATE.feats_dict_tr_2) > max_preloaded_feats:
                    threading._sleep(1.0)
                    if verbose:
                        print ('... bottom part (%d) is waiting for features in the dictionary to get consumed by top part' % (instance_num))

            elif instance_num == 3:
                # wait looping if there are so many features in the dictionary
                while len(TRAIN_STATE.feats_dict_tr_3) > max_preloaded_feats:
                    threading._sleep(1.0)
                    if verbose:
                        print ('... bottom part (%d) is waiting for features in the dictionary to get consumed by top part' % (instance_num))

            # loop on groups according to instances
            if instance_num == 1 and idx_video % n_instances != 0:
                continue

            if instance_num == 2 and idx_video % n_instances != 1:
                continue

            if instance_num == 3 and idx_video % n_instances != 2:
                continue

            tg_1 = time.time()
            video_name = video_names[idx_video]
            video_num = idx_video + 1

            # wait until the image_batch is loaded
            t1 = time.time()
            while video_reader.is_busy():
                threading._sleep(0.1)
            t2 = time.time()
            duration_waited = t2 - t1
            if verbose:
                print('\n... ... model bottom (%d), video %d/%d, waited: %d, name: %s' % (instance_num, video_num, n_videos, duration_waited, video_name))

            # get the frames
            frames = video_reader.get_images()  # (G*T*N, 224, 224, 3)

            # pre-load for the next video group, notice that we take into account the number of instances
            if idx_video + n_instances < n_videos:
                next_video_num = video_num + n_instances
                next_video_name = video_names[idx_video + n_instances]
                next_video_frames = video_frames_dict[next_video_name]
                video_reader.load_imgs_in_batch(next_video_frames)
                if verbose:
                    print('\n... ... model bottom (%d), next video %d/%d, name: %s' % (instance_num, next_video_num, n_videos, next_video_name))

            if video_name in TRAIN_STATE.feats_dict_tr_1 or video_name in TRAIN_STATE.feats_dict_tr_2 or video_name in TRAIN_STATE.feats_dict_tr_3:
                raise ('... ... this should not be happening, but features for video %s already exist in the dictionary' % (video_name))

            if len(frames) != n_frames_per_video:
                raise ('... ... wrong n frames for video: %s' % (video_name))

            # reshape to make one dimension carries the frames / segment, while the other dimesion represents the batch size
            frames = np.reshape(frames, [n_frames_out, n_frames_per_segment, 224, 224, 3])  # (T, 8, 224, 224, 3)

            # get features
            features = model.predict(frames)  # (T, 1, 7, 7, 1024)

            # remove temporal axis, as it is one
            features = np.squeeze(features, axis=1)  # (T, 7, 7, 1024)

            # add feature to the dictionary
            if instance_num == 1:
                TRAIN_STATE.feats_dict_tr_1[video_name] = features
            elif instance_num == 2:
                TRAIN_STATE.feats_dict_tr_2[video_name] = features
            elif instance_num == 3:
                TRAIN_STATE.feats_dict_tr_3[video_name] = features

            tg_2 = time.time()
            if verbose:
                print ('took', tg_2 - tg_1)

        t2 = time.time()
        if verbose:
            print('... finish extracting features in %d seconds' % (t2 - t1))

        # after finishing epoch, update counters
        if instance_num == 1:
            TRAIN_STATE.model_bottom_1_epoch_end = epoch_num
        if instance_num == 2:
            TRAIN_STATE.model_bottom_2_epoch_end = epoch_num
        if instance_num == 3:
            TRAIN_STATE.model_bottom_3_epoch_end = epoch_num

        # wait untill the other part finishes
        if instance_num == 1:
            while TRAIN_STATE.model_bottom_1_epoch_end > TRAIN_STATE.model_bottom_2_epoch_end or TRAIN_STATE.model_bottom_1_epoch_end > TRAIN_STATE.model_bottom_3_epoch_end:
                threading._sleep(1.0)
                if verbose:
                    print ('... bottom part (1) is waiting for bottom part (2,3) to finish extracting features on epoch %d' % (epoch_num))
        if instance_num == 2:
            while TRAIN_STATE.model_bottom_2_epoch_end > TRAIN_STATE.model_bottom_1_epoch_end or TRAIN_STATE.model_bottom_2_epoch_end > TRAIN_STATE.model_bottom_3_epoch_end:
                threading._sleep(1.0)
                if verbose:
                    print ('... bottom part (2) is waiting for bottom part (1,3) to finish extracting features on epoch %d' % (epoch_num))
        if instance_num == 3:
            while TRAIN_STATE.model_bottom_3_epoch_end > TRAIN_STATE.model_bottom_1_epoch_end or TRAIN_STATE.model_bottom_3_epoch_end > TRAIN_STATE.model_bottom_2_epoch_end:
                threading._sleep(1.0)
                if verbose:
                    print ('... bottom part (3) is waiting for bottom part (1,2) to finish extracting features on epoch %d' % (epoch_num))

        # if top part is not finished yet, then wait
        while TRAIN_STATE.model_top_epoch_end < TRAIN_STATE.model_bottom_1_epoch_end or TRAIN_STATE.model_top_epoch_end < TRAIN_STATE.model_bottom_2_epoch_end or TRAIN_STATE.model_top_epoch_end < TRAIN_STATE.model_bottom_3_epoch_end:
            threading._sleep(2.0)
            if verbose:
                print ('... bottom part (%d) is waiting for top part to finish training on epoch: %d' % (instance_num, TRAIN_STATE.model_top_epoch_end + 1))

    print('... finish extracting features for all epochs, goodbye!')
    print('... end time: %s' % utils.timestamp())

# endregion

# region Train Helpers

def __get_features_from_dictionaries(video_names_tr_batch):
    global TRAIN_STATE

    features = []
    for v_name in video_names_tr_batch:
        if v_name in TRAIN_STATE.feats_dict_tr_1:
            features.append(TRAIN_STATE.feats_dict_tr_1[v_name])
        elif v_name in TRAIN_STATE.feats_dict_tr_2:
            features.append(TRAIN_STATE.feats_dict_tr_2[v_name])
        elif v_name in TRAIN_STATE.feats_dict_tr_3:
            features.append(TRAIN_STATE.feats_dict_tr_3[v_name])
        else:
            raise Exception('This should not be happening, but a feature is asked for and it does not exist in the dictionaries: %s' % (v_name))

    return features

def __shuffle_training_data(video_names_tr, class_nums_tr):
    n_v = len(video_names_tr)
    idx = np.arange(n_v)
    random.shuffle(idx)
    video_names_tr = video_names_tr[idx]
    class_nums_tr = class_nums_tr[idx]

    return video_names_tr, class_nums_tr

def __save_model_backbone(root_model, model_name, epoch_num):
    model_root_path = Pth('EPIC-Kitchens/models_backbone/%s', (model_name,))
    if not os.path.exists(model_root_path):
        os.mkdir(model_root_path)

    model_path = '%s/%03d.model' % (model_root_path, epoch_num)
    model_json_path = '%s/%03d.json' % (model_root_path, epoch_num)
    model_weight_path = '%s/%03d.pkl' % (model_root_path, epoch_num)

    # for very long model, this does not work
    # self.root_model.save(model_path)
    # only save model definition and weights
    keras_utils.save_model(root_model, model_json_path, model_weight_path)

def __save_model(root_model, model_name, epoch_num):
    model_root_path = Pth('EPIC-Kitchens/models/%s' % (model_name,))
    if not os.path.exists(model_root_path):
        os.mkdir(model_root_path)

    model_path = '%s/%03d.model' % (model_root_path, epoch_num)
    model_json_path = '%s/%03d.json' % (model_root_path, epoch_num)
    model_weight_path = '%s/%03d.pkl' % (model_root_path, epoch_num)

    # for very long model, this does not work
    # self.root_model.save(model_path)
    # only save model definition and weights
    keras_utils.save_model(root_model, model_json_path, model_weight_path)

def __save_centroids(root_model, model_name, epoch_num):
    centroids_root_path = Pth('EPIC-Kitchens/node_features/%s', (model_name,))
    centroids_path = '%s/%03d.pkl' % (centroids_root_path, epoch_num)

    if not os.path.exists(centroids_root_path):
        os.mkdir(centroids_root_path)

    session = K.get_session()
    t_centroids = root_model.get_layer('node_embedding').output  # (1, 20, 1024)
    centroids_embedding = t_centroids.eval(session=session)  # (1, 20, 1024)
    centroids_embedding = np.squeeze(centroids_embedding, axis=0)

    utils.pkl_dump(centroids_embedding, centroids_path)

def __compile_model_for_finetuning(model, n_gpus):
    # optimizer and loss
    loss = keras_utils.LOSSES[3]
    optimizer = Adam(lr=0.01, epsilon=1e-8)
    optimizer = Adam(lr=0.001, epsilon=1e-4)
    optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0000001)
    optimizer = SGD(lr=0.02, momentum=0.8)

    if n_gpus == 1:
        model.compile(loss=loss, optimizer=optimizer)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer)

    return model, parallel_model

# endregion

# region Models

def __load_model_timeception(n_classes, input_shape, n_gpus, is_load_weights, weight_path):
    """
    Model
    """
    # optimizer and loss
    loss = keras_utils.LOSSES[3]
    output_activation = keras_utils.ACTIVATIONS[2]
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
        model.compile(loss=loss, optimizer=optimizer)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer)

    return model, parallel_model

def __load_model_action_vlad(n_classes, input_shape, n_gpus, is_load_weights, weight_path):
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
        model.compile(loss=loss, optimizer=optimizer)
        parallel_model = model
    else:
        parallel_model = multi_gpu_utils.multi_gpu_model(model, n_gpus)
        parallel_model.compile(loss=loss, optimizer=optimizer)

    return model, parallel_model

def __load_model_videograph(nodes, n_classes, input_shape_x):
    """
    Model
    """

    # optimizer and loss
    loss = keras_utils.LOSSES[3]
    output_activation = keras_utils.ACTIVATIONS[2]
    optimizer = SGD(lr=0.01)
    optimizer = Adam(lr=0.01, epsilon=1e-8)
    optimizer = Adam(lr=0.01, epsilon=1e-4)

    # per-layer kernel size and max pooling for nodes and timesteps
    n_graph_layers = 2

    # time kernel
    t_kernel_size = 7
    t_max_size = 3

    # node kernel
    n_kernel_size = 7
    n_max_size = 3
    n_avg_size = 4

    # space kernel
    s_kernel_size = 2
    s_kernel_size = 1

    n_nodes, _ = nodes.shape

    _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape_x
    t_input_x = Input(shape=(n_timesteps, side_dim, side_dim, n_channels_in), name='input_x')  # (None, 64, 1024)
    t_input_n = Input(tensor=tf.constant(nodes, dtype=tf.float32), name='input_n')  # (1, 100, 1024)
    tensor = t_input_x

    # spatial convolution
    tensor = Conv3D(n_channels_in, (1, s_kernel_size, s_kernel_size), padding='VALID', name='conv_s')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)

    # pool over space
    tensor = MaxLayer(axis=(2, 3), is_keep_dim=True, name='global_pool_s')(tensor)  # (None, 64, 7, 7, 1024)

    # node attention
    tensor = videograph.node_attention(tensor, t_input_n, n_channels_in, activation_type='relu')  # (N, 100, 64, 7, 7, 1024)

    # graph embedding
    tensor = videograph.graph_embedding(tensor, n_graph_layers, n_avg_size, n_kernel_size, t_kernel_size, n_max_size, t_max_size)  # (N, 100, 64, 7, 7, 1024)

    # node pooling
    tensor = MeanLayer(axis=(1,), name='global_pool_n')(tensor)

    # temporal pooling
    tensor = MaxLayer(axis=(1, 2, 3), name='global_pool_t')(tensor)

    # mlp for classification
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(512)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(n_classes)(tensor)
    t_output = Activation(output_activation)(tensor)

    model = Model(input=[t_input_x, t_input_n], output=t_output)
    model.compile(loss=loss, optimizer=optimizer)
    return model

# endregion

# region Functions

def __load_annotation(annotation_path, annotation_type):
    annotation_types = ['noun', 'verb', 'noun_verb', 'action']
    assert annotation_type in annotation_types

    (y_noun_tr, y_verb_tr, y_actn_tr, y_noun_te, y_verb_te, y_actn_te) = utils.pkl_load(annotation_path)
    if annotation_type == 'noun':
        n_classes = ds_epic_kitchens.N_NOUNS_MANY_SHOT
        (y_tr, y_te) = (y_noun_tr, y_noun_te)
    elif annotation_type == 'verb':
        n_classes = ds_epic_kitchens.N_VERBS_MANY_SHOT
        (y_tr, y_te) = (y_verb_tr, y_verb_te)
    elif annotation_type == 'noun_verb':
        n_classes = ds_epic_kitchens.N_NOUNS_MANY_SHOT + ds_epic_kitchens.N_VERBS_MANY_SHOT
        (y_tr, y_te) = (np.hstack((y_noun_tr, y_verb_tr)), np.hstack((y_noun_te, y_verb_te)))
    elif annotation_type == 'action':
        n_classes = ds_epic_kitchens.N_ACTNS_MANY_SHOT
        (y_tr, y_te) = (y_actn_tr, y_actn_te)
    else:
        raise Exception('Sorry, unknown annotation type: %s' % (annotation_type))

    return (y_tr, y_te), n_classes

# endregion

# region Classes

class TrainingState(object):
    """
    An instance of this object is serialized and saved in the features path. This object is read by both top and bottom layer, as a way for communication.
    """

    def __init__(self):
        self.model_top_epoch_start = 0
        self.model_top_epoch_end = 0
        self.model_bottom_1_epoch_start = 0
        self.model_bottom_2_epoch_start = 0
        self.model_bottom_3_epoch_start = 0
        self.model_bottom_1_epoch_end = 0
        self.model_bottom_2_epoch_end = 0
        self.model_bottom_3_epoch_end = 0
        self.epoch_num_model_top = 0
        self.model_top_ready = False
        self.video_names_tr = None
        self.class_nums_tr = None
        self.video_frames_dict_tr = {}
        self.feats_dict_tr_1 = {}
        self.feats_dict_tr_2 = {}
        self.feats_dict_tr_3 = {}

# endregion
