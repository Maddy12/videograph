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
# 1.0 First, experiments on Breakfast
########################################################################
########################################################################

#########################
# 1.1 Prepare Data
#########################
# If during the taining you find any missing annotation file, please search in 'ds_breakfast.py' and seen how to re-generate it.
# We extract features using I3D model, pre-trained on Kinetics.
ds_breakfast._301_extract_features_i3d()
# Generate initial values of centroids, once and for all
ds_breakfast._501_generate_centroids(n_centroids=32, n_dims=1024)
ds_breakfast._501_generate_centroids(n_centroids=64, n_dims=1024)
ds_breakfast._501_generate_centroids(n_centroids=128, n_dims=1024)

#########################
# 1.2 Train Models
#########################
# We train the model (i.e. Video_Graph) using the previously extracted features
# You have to options, either to pickle the features in one big hdf5 file, and load it and train on its feature
ds_breakfast._401_pickle_features_i3d_mixed_5c()
exp_breakfast_activities.train_model_on_pickled_features()
# Or train on features, where the features of each video is pickled in one pkl file, and loaded on demand for each batch
exp_breakfast_activities.train_model_on_video_features_i3d()

#########################
# 1.3 Visualize Results
#########################
an_breakfast


########################################################################
# 2.0 Second, experiments on Charades
########################################################################
# train videograph on features of Charades
# from experiments import Charades
# Charades.train_model_videograph()


########################################################################
########################################################################
# 3.0 Second, experiments on Epic-Kitchens
########################################################################
########################################################################

# train videograph on features of Epic_Kitchens
# from experiments import Epic_Kitchens
# Epic_Kitchens.train_model_videograph()