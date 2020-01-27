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

import numpy as np
import platform

DL_FRAMEWORK = None
GPU_CORE_ID = 0
PLATFORM = None
MACHINE_NAME = platform.node()

DL_FRAMEWORKS = np.array(['caffe', 'tensorflow', 'pytorch', 'keras', 'caffe2'])
CNN_MODEL_TYPES = np.array(['resnet152', 'googlenet1k', 'vgg16', 'places365-resnet152', 'places365-vgg', 'googlenet13k'])
CNN_FEATURE_TYPES = np.array(['fc6', 'fc7', 'fc1000', 'fc1024', 'fc365', 'prob', 'pool5', 'fc8a', 'res3b7', 'res4b35', 'res5c'])
CNN_FEATURE_SIZES = np.array([2048, 2048, 1000, 1024, 1000, 2048, 2048])
SIMILARITY_TYPES = np.array(['cosine', 'kl', 'euclidean'])
POOLING_TYPES = np.array(['avg', 'norm_max', 'max'])
MANIFOLD_TYPES = np.array(['tsne', 'isomap', 'mds', 'spectral'])
RESIZE_TYPES = np.array(['resize', 'resize_crop', 'resize_crop_scaled', 'resize_keep_aspect_ratio_padded'])

# replace this directory with the directory of the data folder in this project, for example
DATA_ROOT_PATH = '/home/nour/Documents/PyCharmProjects/video_graph/data'
# replace this directory with the directory of this project, for example
PROJECT_ROOT_PATH = '/home/nour/Documents/PyCharmProjects/video_graph'
