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
Main file of the project.
"""

from datasets import ds_breakfast, ds_charades, ds_epic_kitchens
from experiments import exp_breakfast_activities, exp_breakfast_unit_actions, exp_epic_kitchens, exp_charades
from analysis import an_breakfast

# Please follow the following steps to reproduce the results of the paper.

########################################################################
########################################################################
# 0.0 Directory Preparations
########################################################################
########################################################################
# 0.1 Preparing the projects
# Make sure the root directories are correctly modified
# core.configs.DATA_ROOT_PATH = '???'
# core.configs.PROJECT_ROOT_PATH = '????'

########################################################################
########################################################################
# 1.0 Experiments on Breakfast: Single-label Classification (i.e. Activities)
########################################################################
########################################################################

#########################
# 1.1 Prepare Data
#########################
# extract frames from videos
ds_breakfast._201_extract_frames()
# If during the taining you find any missing annotation file, please search in 'ds_breakfast.py' and seen how to re-generate it.
# We extract features using I3D model, pre-trained on Kinetics.
ds_breakfast._301_extract_features_i3d()
# Generate initial values of centroids, once and for all
ds_breakfast._501_generate_centroids(n_centroids=128, n_dims=1024)

#########################
# 1.2 Train Models
#########################
# We train the model (i.e. Video_Graph) using the previously extracted features
# You have to options, either to pickle the features in one big hdf5 file, and load it and train on its feature
ds_breakfast._401_pickle_features_i3d_mixed_5c()
exp_breakfast_activities.train_model_on_pickled_features()
# Or train on features, where the features of each video is pickled in one pkl file, and loaded on demand for each batch
# make dure to edit the model trained
exp_breakfast_activities.train_model_on_video_features_i3d()

#########################
# 1.3 Visualize Results
#########################
# here, you can find code to reproduce the graph diagram, see figure 7 in the paper.
an_breakfast._07_visualize_graph_edges()

########################################################################
########################################################################
# 2.0 Experiments on Breakfast: Multi-label Classification (i.e. Unit-actions)
########################################################################
########################################################################

#########################
# 2.2 Train Models
#########################
exp_breakfast_unit_actions.train_model_on_pickled_features()

########################################################################
########################################################################
# 3.0 Experiments on Epic-Kitchens
########################################################################
########################################################################

#########################
# 3.1 Prepare Data
#########################

# prepare some annotations
ds_epic_kitchens._101_prepare_annot_id_of_many_shots()
ds_epic_kitchens._102_prepare_data_splits()
ds_epic_kitchens._103_prepare_many_shots_noun_verb_action_ids()

# relative pathes of frames
ds_epic_kitchens._201_prepare_video_frames_path_dict()
ds_epic_kitchens._202_spit_video_frames_relative_pathes()

# sample frames from the videos
ds_epic_kitchens._301_sample_frame_pathes_i3d()

# extract frames from videos
ds_epic_kitchens._401_extract_features_i3d()
ds_epic_kitchens._501_pickle_features_i3d()

# generate centroids
ds_epic_kitchens._602_generate_nodes(128, 1024)

#########################
# 3.2 Train Models
#########################

# train videograph on features of Epic_Kitchens.
# this gets you inferior results, as it uses the same sampled frames throughtout the entire training
exp_epic_kitchens.train_model_on_pickled_features()

# but to reproduce the results, you need to sample frames every epoch, for this, you use this
exp_epic_kitchens.train_model_on_video_frames()

# if you want even better results (which we did NOT consider in our paper), you can fine-tuned I3D on Epic-Kitchens
# but you have to implement this by your self

########################################################################
# 4.0 Experiments on Charades
########################################################################
# train videograph on features of Charades
# from experiments import ds_charades
