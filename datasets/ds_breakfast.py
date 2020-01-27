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
We have 1712 videos
We have 52 persons/actors
10 activities
48 action units
persons: 52 total, 44 train, 8 test
videos: 1712 total, 1357 train, 335 test
"""

import os
import numpy as np
import time
import cv2
import natsort
import threading

from core import utils, video_utils, sobol
from core.utils import Path as Pth
from core import const as c

from nets.resnet_152_keras import ResNet152
from nets.i3d_keras import Inception_Inflated3d
from core.image_utils import AsyncImageReaderBreakfastForI3DKerasModel

# region Constants

N_CLASSES_ACTIONS = 48
N_CLASSES_ACTIVITIES = 10

# endregion

# region 1.0 Prepare Annotation

def _101_prepare_action_ids():
    """
    Get list of all unit-actions and activities
    :return:
    """

    video_types = ['cam', 'webcam', 'stereo']

    videos_root_path = Pth('Breakfast/videos')
    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.pkl')
    activities_path = Pth('Breakfast/annotation/activities_list.pkl')

    person_names = utils.folder_names(videos_root_path, is_nat_sort=True)

    unit_actions = []
    activities = []

    # loop on persons
    for person_name in person_names:
        p_video_root_path = '%s/%s' % (videos_root_path, person_name)

        p_video_types = [n for n in utils.folder_names(p_video_root_path) if __check_correct_video_type(video_types, n)]
        p_video_types = np.array(p_video_types)

        # loop on videos for each person
        for p_video_type in p_video_types:
            # get file names
            instance_video_root_path = '%s/%s' % (p_video_root_path, p_video_type)
            instance_video_names = utils.file_names(instance_video_root_path, is_nat_sort=True)

            # if stereo videos, consider only the first channel
            instance_video_names = [n for n in instance_video_names if utils.get_file_extension(n) == 'avi' and ('stereo' not in p_video_type or 'ch0' in n)]

            # append relative pathes of videos
            instance_video_relative_pathes = ['Breakfast/videos/%s/%s/%s' % (person_name, p_video_type, n) for n in instance_video_names]

            # also, get ground truth for unit-actions and activities
            instance_annot_file_pathes = ['%s/%s.txt' % (instance_video_root_path, utils.remove_extension(n)) for n in instance_video_names]
            instance_unit_actions = __get_action_names_from_files(instance_annot_file_pathes)
            instance_activities = [utils.remove_extension(n).split('_')[1] for n in instance_video_relative_pathes]

            unit_actions += instance_unit_actions
            activities += instance_activities

    activities = np.unique(activities)
    activities = natsort.natsorted(activities)
    activities = np.array(activities)

    unit_actions = np.unique(unit_actions)
    unit_actions = natsort.natsorted(unit_actions)
    unit_actions = np.array(unit_actions)

    print len(activities), len(unit_actions)
    print activities
    print unit_actions

    utils.pkl_dump(unit_actions, unit_actions_path)
    utils.pkl_dump(activities, activities_path)

    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.txt')
    activities_path = Pth('Breakfast/annotation/activities_list.txt')
    utils.txt_dump(unit_actions, unit_actions_path)
    utils.txt_dump(activities, activities_path)

def _102_prepare_video_annot():
    """
    Check ground truth of each video.
    :return:
    """

    video_types = ['cam', 'webcam', 'stereo']

    videos_root_path = Pth('Breakfast/videos')
    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.pkl')
    activities_path = Pth('Breakfast/annotation/activities_list.pkl')

    annot_unit_actions_path = Pth('Breakfast/annotation/annot_unit_actions.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')

    unit_actions = utils.pkl_load(unit_actions_path)
    activities = utils.pkl_load(activities_path)
    person_names = utils.folder_names(videos_root_path, is_nat_sort=True)

    split_ratio = 0.85
    video_relative_pathes_tr = []
    video_relative_pathes_te = []

    y_unit_actions_tr = []
    y_unit_actions_te = []

    y_activities_tr = []
    y_activities_te = []

    n_persons = len(person_names)
    n_persons_tr = int(n_persons * split_ratio)
    n_persons_te = n_persons - n_persons_tr
    person_names_tr = person_names[:n_persons_tr]
    person_names_te = person_names[n_persons_tr:]

    # loop on persons
    for person_name in person_names:
        p_video_root_path = '%s/%s' % (videos_root_path, person_name)

        p_video_types = [n for n in utils.folder_names(p_video_root_path) if __check_correct_video_type(video_types, n)]
        p_video_types = np.array(p_video_types)

        # loop on videos for each person
        for p_video_type in p_video_types:
            # get file names
            instance_video_root_path = '%s/%s' % (p_video_root_path, p_video_type)
            instance_video_names = utils.file_names(instance_video_root_path, is_nat_sort=True)

            # if stereo videos, consider only the first channel
            instance_video_names = [n for n in instance_video_names if utils.get_file_extension(n) == 'avi' and ('stereo' not in p_video_type or 'ch0' in n)]

            # append relative pathes of videos
            instance_video_relative_pathes = ['%s/%s/%s' % (person_name, p_video_type, n) for n in instance_video_names]

            # also, get ground truth for unit-actions and activities
            instance_activities_y, instance_unit_actions_y = __get_gt_activities_and_actions(instance_video_root_path, instance_video_names, activities, unit_actions)

            if person_name in person_names_tr:
                video_relative_pathes_tr += instance_video_relative_pathes
                y_unit_actions_tr += instance_unit_actions_y
                y_activities_tr += instance_activities_y
            else:
                video_relative_pathes_te += instance_video_relative_pathes
                y_unit_actions_te += instance_unit_actions_y
                y_activities_te += instance_activities_y

    video_relative_pathes_tr = np.array(video_relative_pathes_tr)
    video_relative_pathes_te = np.array(video_relative_pathes_te)

    y_activities_tr = np.array(y_activities_tr)
    y_activities_te = np.array(y_activities_te)

    y_unit_actions_tr = np.array(y_unit_actions_tr)
    y_unit_actions_te = np.array(y_unit_actions_te)

    print video_relative_pathes_tr.shape
    print video_relative_pathes_te.shape

    print y_activities_tr.shape
    print y_activities_te.shape

    print y_unit_actions_tr.shape
    print y_unit_actions_te.shape

    # finally, save video annotation ()
    annot_unit_action = (video_relative_pathes_tr, y_unit_actions_tr, video_relative_pathes_te, y_unit_actions_te)
    annot_activities = (video_relative_pathes_tr, y_activities_tr, video_relative_pathes_te, y_activities_te)
    utils.pkl_dump(annot_unit_action, annot_unit_actions_path)
    utils.pkl_dump(annot_activities, annot_activities_path)

    return

def _103_prepare_video_info():
    video_info_path = Pth('Breakfast/annotation/video_info.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_relative_pathes = np.hstack((video_relative_pathes_tr, video_relative_pathes_te))
    n_videos = len(video_relative_pathes)

    video_info = dict()
    fps, n_frames, duration = [], [], []

    # loop on the videos
    for idx_video, video_relative_path in enumerate(video_relative_pathes):

        utils.print_counter(idx_video, n_videos, 100)

        video_path = Pth('Breakfast/videos/%s', (video_relative_path,))
        video_id = __video_relative_path_to_video_id(video_relative_path)

        try:
            v_fps, v_n_frames, v_duration = video_utils.get_video_info(video_path)
        except:
            print video_relative_path
            continue

        fps.append(v_fps)
        n_frames.append(v_n_frames)
        duration.append(v_duration)
        video_info[video_id] = {'duration': v_duration, 'fps': v_fps, 'n_frames': v_n_frames}

    print np.mean(fps), np.std(fps), np.min(fps), np.max(fps)
    print np.mean(duration), np.std(duration), np.min(duration), np.max(duration)
    print np.mean(n_frames), np.std(n_frames), np.min(n_frames), np.max(n_frames)

    # 15.0 0.0 15.0 15.0
    # 140.30865654205607 121.76493338896255 12.4 649.67
    # 2105.308995327103 1826.5189539717755 187 9746

    utils.pkl_dump(video_info, video_info_path)

def _104_prepare_video_gt():
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    annot_actions_path = Pth('Breakfast/annotation/annot_unit_actions.pkl')
    gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions.pkl')
    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)

    (video_relative_pathes_tr, annot_activities_tr, video_relative_pathes_te, annot_activities_te) = utils.pkl_load(annot_activities_path)
    video_relative_pathes_tr = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_tr])
    video_relative_pathes_te = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_te])

    gt_activities_tr = []
    gt_activities_te = []

    gt_actions_tr = []
    gt_actions_te = []

    for video_id in video_ids_tr:
        idx = np.where(video_id == video_relative_pathes_tr)[0][0]
        gt_activities_tr.append(annot_activities_tr[idx])

    for video_id in video_ids_te:
        idx = np.where(video_id == video_relative_pathes_te)[0][0]
        gt_activities_te.append(annot_activities_te[idx])

    (video_relative_pathes_tr, annot_actions_tr, video_relative_pathes_te, annot_actions_te) = utils.pkl_load(annot_actions_path)
    video_relative_pathes_tr = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_tr])
    video_relative_pathes_te = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_te])

    for video_id in video_ids_tr:
        idx = np.where(video_id == video_relative_pathes_tr)[0][0]
        gt_actions_tr.append(annot_actions_tr[idx])

    for video_id in video_ids_te:
        idx = np.where(video_id == video_relative_pathes_te)[0][0]
        gt_actions_te.append(annot_actions_te[idx])

    gt_activities_tr = np.array(gt_activities_tr)
    gt_activities_te = np.array(gt_activities_te)
    gt_actions_tr = np.array(gt_actions_tr)
    gt_actions_te = np.array(gt_actions_te)

    print gt_activities_tr.shape
    print gt_activities_te.shape
    print gt_actions_tr.shape
    print gt_actions_te.shape

    utils.pkl_dump(((video_ids_tr, gt_activities_tr), (video_ids_te, gt_activities_te)), gt_activities_path)
    utils.pkl_dump(((video_ids_tr, gt_actions_tr), (video_ids_te, gt_actions_te)), gt_actions_path)

def _105_prepare_action_gt_timestamped():
    """
    Get ground truth of unit-actions with their timestamps.
    :return:
    """
    root_path = c.DATA_ROOT_PATH
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.pkl')
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions_timestamped.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)
    unit_actions = utils.pkl_load(unit_actions_path)

    video_pathes_tr = ['%s/Breakfast/videos/%s' % (root_path, __video_video_id_to_video_relative_path(id, False),) for id in video_ids_tr]
    video_pathes_te = ['%s/Breakfast/videos/%s' % (root_path, __video_video_id_to_video_relative_path(id, False),) for id in video_ids_te]

    gt_actions_te = __get_gt_actions_timestamped(video_pathes_te, unit_actions)
    gt_actions_tr = __get_gt_actions_timestamped(video_pathes_tr, unit_actions)

    gt_actions_tr = np.array(gt_actions_tr)
    gt_actions_te = np.array(gt_actions_te)

    l_tr = [len(i) for i in gt_actions_tr]
    l_te = [len(i) for i in gt_actions_te]
    print ('mean, std, min, max for number of nodes in each video [tr/te]')
    print np.mean(l_tr), np.std(l_tr), np.min(l_tr), np.max(l_tr)
    print np.mean(l_te), np.std(l_te), np.min(l_te), np.max(l_te)

    print gt_actions_tr.shape
    print gt_actions_te.shape

    utils.pkl_dump(((video_ids_tr, gt_actions_tr), (video_ids_te, gt_actions_te)), gt_actions_path)

def _106_prepare_action_graph_vector():
    """
    Each video is labled with a set of actions, we construct a graph using these actions.
    Links represent the relationship between two nodes. A node however represents one action.
    For a video, a link is only drawn between two nodes if these two nodes are neighbours.
    :return:
    """

    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions_timestamped.pkl')
    action_graph_vectors_path = Pth('Breakfast/annotation/action_graph_vectors.pkl')
    action_graph_matrices_path = Pth('Breakfast/annotation/action_graph_matrices.pkl')
    (video_ids_tr, gt_actions_tr), (video_ids_te, gt_actions_te) = utils.pkl_load(gt_actions_path)

    graph_matrices_tr = __get_action_graph_matrices(video_ids_tr, gt_actions_tr)
    graph_matrices_te = __get_action_graph_matrices(video_ids_te, gt_actions_te)

    graph_vectors_tr = __get_action_graph_vectors(video_ids_tr, gt_actions_tr)
    graph_vectors_te = __get_action_graph_vectors(video_ids_te, gt_actions_te)

    print graph_matrices_tr.shape
    print graph_matrices_te.shape
    print graph_vectors_tr.shape
    print graph_vectors_te.shape

    # save the graph data
    utils.pkl_dump((graph_matrices_tr, graph_matrices_te), action_graph_matrices_path)
    utils.pkl_dump((graph_vectors_tr, graph_vectors_te), action_graph_vectors_path)

def __get_action_names_from_files(pathes):
    action_names = []

    for path in pathes:
        lines = utils.txt_load(path)
        for l in lines:
            action_name = l.split(' ')[1]
            action_names.append(action_name)

    return action_names

def __get_gt_activities_and_actions(root_path, video_names, activities, unit_actions):
    y_activities = []
    y_actions = []

    for video_name in video_names:
        # first, get idx of activity
        activity = utils.remove_extension(video_name).split('_')[1]
        idx_activity = np.where(activity == activities)[0][0]
        y_activity = np.zeros((N_CLASSES_ACTIVITIES,), dtype=np.int)
        y_activity[idx_activity] = 1
        y_activities.append(y_activity)

        # then, get idx of actions
        action_txt_path = '%s/%s.txt' % (root_path, utils.remove_extension(video_name))
        lines = utils.txt_load(action_txt_path)
        idx_actions = [np.where(unit_actions == l.split(' ')[1])[0][0] for l in lines]
        y_action = np.zeros((N_CLASSES_ACTIONS,), dtype=np.int)
        y_action[idx_actions] = 1
        y_actions.append(y_action)

    return y_activities, y_actions

def __get_gt_actions_timestamped(video_pathes, unit_actions):
    y_actions = []

    for video_path in video_pathes:
        # then, get idx of actions
        action_txt_path = '%s.txt' % (video_path)
        lines = utils.txt_load(action_txt_path)

        video_annot = []
        for l in lines:
            line_splits = l.split(' ')
            idx_action = np.where(unit_actions == line_splits[1])[0][0]
            frame_start, frame_end = line_splits[0].split('-')
            frame_start = int(frame_start)
            frame_end = int(frame_end)
            video_annot.append((idx_action, frame_start, frame_end))

        y_actions.append(video_annot)

    return y_actions

def __check_correct_video_type(video_types, n):
    for t in video_types:
        if t in n:
            return True
    return False

def __video_relative_path_to_video_id(relative_path):
    video_id = utils.remove_extension(relative_path).replace('/', '_')
    return video_id

def __video_video_id_to_video_relative_path(id, include_extension=True):
    splits = tuple(id.split('_'))
    s_format = '%s/%s/%s_%s' if len(splits) == 4 else '%s/%s/%s_%s_%s'
    video_path = s_format % splits
    video_path = '%s.avi' % video_path if include_extension else video_path
    return video_path

def __get_action_graph_matrices(video_ids, gt_action_timestamped):
    n_videos = len(video_ids)
    n_actions = N_CLASSES_ACTIONS
    n_neighnours = 2

    graph_matrices = np.zeros((n_videos, n_actions, n_actions), dtype=np.int)

    # loop on all videos
    for idx_video, video_id in enumerate(video_ids):

        # matrix to save distances
        graph_matrix = np.zeros((n_actions, n_actions), dtype=np.int)

        # get annotation of certain video
        video_action_labels = gt_action_timestamped[idx_video]

        n_labels = len(video_action_labels)
        n_local_windows = n_labels - n_neighnours

        for idx in range(n_local_windows):

            # get all items inside this local window, items can be either: verbs, nouns or actions
            local_action_labels = video_action_labels[idx:idx + n_neighnours]
            local_ids = np.array([l[0] for l in local_action_labels])

            # add the distances to the matrix, distances are only in this local window
            for i in range(n_neighnours):
                for j in range(i + 1, n_neighnours):
                    id_1 = local_ids[i]
                    id_2 = local_ids[j]

                    # if two nodes are the same, then don't consider
                    if id_1 == id_2:
                        continue
                        # set value = 1 to denote a link
                    graph_matrix[id_1, id_2] = 1

        # add the current matrix to list of matrices
        graph_matrices[idx_video] = graph_matrix

    return graph_matrices

def __get_action_graph_vectors(video_ids, gt_action_timestamped):
    n_actions = N_CLASSES_ACTIONS
    n_videos = len(video_ids)

    # if we have n nouns, then to save all pairwise distances between nouns, we need (n-1)*(n/2) values
    vector_dim = n_actions * (n_actions - 1) * 0.5
    assert vector_dim - int(vector_dim) == 0
    vector_dim = int(vector_dim)
    graph_vectors = np.zeros((n_videos, vector_dim), dtype=np.int)
    idx_matrix = __get_idx_matrix(n_actions)

    # number of neighbours in a local window
    n_neighnours = 2

    # loop on all videos
    for idx_video, video_id in enumerate(video_ids):

        graph_vector = np.zeros((vector_dim,), dtype=np.int)

        # get annotation of certain video
        video_action_labels = gt_action_timestamped[idx_video]

        n_labels = len(video_action_labels)
        n_local_windows = n_labels - n_neighnours

        for idx in range(n_local_windows):

            # get all items inside this local window, items can be either: verbs, nouns or actions
            local_action_labels = video_action_labels[idx:idx + n_neighnours]
            local_ids = np.array([l[0] for l in local_action_labels])

            # add the distances to the matrix, distances are only in this local window
            for i in range(n_neighnours):
                for j in range(i + 1, n_neighnours):
                    id_1 = local_ids[i]
                    id_2 = local_ids[j]
                    # if two nodes are different
                    if id_1 == id_2:
                        continue

                    # add value = 1 to denote a link between current two nodes, i.e. two actions
                    id_vector = idx_matrix[id_1, id_2]
                    graph_vector[id_vector] = 1

        # append the distance_vector
        graph_vectors[idx_video] = graph_vector

    # save distance matrix of the video
    return graph_vectors

def __get_idxes_to_convert_graph_matrix_to_vector(n_dims):
    """
    For a square matrix of size n, we return two lists, each pair in the two lists represent a position in the matrix and it's mirror.
    For example, let n = 3, then it is a 3x3 matrix. Then, the two lists are:
    List1 = [(0, 1), (0, 2), (1, 2)]
    list2 = [(1, 0), (2, 0), (2, 1)]
    :return:
    """

    idxes = []

    for i in range(n_dims):
        for j in range(i, n_dims):
            if j > i:
                idxes.append((i, j))

    idxes = np.array(idxes)
    return idxes

def __get_idx_matrix(n_ids_dict):
    idx_matrix = - 1 * np.ones((n_ids_dict, n_ids_dict), dtype=int)
    idx = -1
    for i in range(n_ids_dict):
        for j in range(n_ids_dict):
            if j > i:
                idx += 1
                idx_matrix[i, j] = idx
                idx_matrix[j, i] = idx

    return idx_matrix

# endregion

# region 2.0 Sample Frames

def _201_extract_frames(begin_num, end_num):
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_relative_pathes = np.hstack((video_relative_pathes_tr, video_relative_pathes_te))
    n_videos = len(video_relative_pathes)

    image_name_format = '%s/%06d.jpg'

    for idx_video, video_relative_path in enumerate(video_relative_pathes):

        if idx_video < begin_num or idx_video >= end_num:
            continue

        t1 = time.time()
        video_id = __video_relative_path_to_video_id(video_relative_path)
        video_path = Pth('Breakfast/videos/%s', (video_relative_path))

        # path to to store video frames
        video_frames_root_path = Pth('Breakfast/frames/%s', (video_id))
        if not os.path.exists(video_frames_root_path):
            os.mkdir(video_frames_root_path)

        # save all frames to disc
        video_utils.video_save_frames(video_path, video_frames_root_path, image_name_format, c.RESIZE_TYPES[1])
        t2 = time.time()
        duration = t2 - t1
        print ('%03d/%03d, %d sec' % (idx_video + 1, end_num, duration))

def _202_sample_frames_i3d():
    """
    Uniformly sample sequences of frames form each video. Each sequences consists of 8 successive frames.
    """
    n_frames_per_video = 512
    n_frames_per_video = 256
    n_frames_per_video = 128
    n_frames_per_video = 1024
    model_type = 'i3d'

    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames_per_video,))

    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_frames_dict_tr = __sample_frames(video_relative_pathes_tr, n_frames_per_video, model_type)
    video_frames_dict_te = __sample_frames(video_relative_pathes_te, n_frames_per_video, model_type)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), frames_annot_path)

def _203_sample_frames_resnet():
    """
    Get list of frames from each video. With max 600 of each video and min 96 frames from each video.
    These frames will be used to extract features for each video.
    """

    # if required frames per video are 128, there are 51/6 out of 7986/1864 videos in training/testing splits that don't satisfy this
    n_frames_per_video = 64
    model_type = 'resnet'

    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_resnet_%d.pkl', (n_frames_per_video,))

    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_frames_dict_tr = __sample_frames(video_relative_pathes_tr, n_frames_per_video, model_type)
    video_frames_dict_te = __sample_frames(video_relative_pathes_te, n_frames_per_video, model_type)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), frames_annot_path)

def _204_sample_frames_non_local():
    """
    Uniformly sample sequences of frames form each video. Each sequences consists of 8 successive frames.
    """

    n_frames_per_video = 512
    model_type = 'non_local'

    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_non_local_%d.pkl', (n_frames_per_video,))

    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_frames_dict_tr = __sample_frames(video_relative_pathes_tr, n_frames_per_video, model_type)
    video_frames_dict_te = __sample_frames(video_relative_pathes_te, n_frames_per_video, model_type)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), frames_annot_path)

def __sample_frames(video_relative_pathes, n_frames_per_video, model_type):
    video_frames_dict = dict()
    n_videos = len(video_relative_pathes)

    assert model_type in ['resnet', 'i3d', 'non_local']

    for idx_video, video_relative_path in enumerate(video_relative_pathes):
        utils.print_counter(idx_video, n_videos, 100)
        video_id = __video_relative_path_to_video_id(video_relative_path)

        # get all frames of the video
        frames_root_path = Pth('Breakfast/frames/%s', (video_id,))
        video_frame_names = utils.file_names(frames_root_path, is_nat_sort=True)

        # sample from these frames
        if model_type == 'resnet':
            video_frame_names = __sample_frames_for_resnet(video_frame_names, n_frames_per_video)
        elif model_type == 'i3d':
            video_frame_names = __sample_frames_for_i3d(video_frame_names, n_frames_per_video)
        elif model_type == 'non_local':
            video_frame_names = __sample_frames_for_non_local(video_frame_names, n_frames_per_video)
        else:
            raise Exception('Unkonwn model type: %s' % (model_type))
        n_frames = len(video_frame_names)
        assert n_frames == n_frames_per_video

        video_frames_dict[video_id] = video_frame_names

    return video_frames_dict

def __sample_frames_for_i_dont_know(frames, n_required):
    # get n frames
    n_frames = len(frames)

    if n_frames < n_required:
        repeats = int(n_required / float(n_frames)) + 1
        idx = np.arange(0, n_frames).tolist()
        idx = idx * repeats
        idx = idx[:n_required]
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        start_idx = int((n_frames - n_required) / 2.0)
        stop_idx = start_idx + n_required
        idx = np.arange(start_idx, stop_idx)

    sampled_frames = np.array(frames)[idx]
    assert len(idx) == n_required
    assert len(sampled_frames) == n_required
    return sampled_frames

def __sample_frames_for_resnet(frames, n_required):
    # get n frames from all over the video
    n_frames = len(frames)

    if n_frames < n_required:
        step = n_frames / float(n_required)
        idx = np.arange(0, n_frames, step, dtype=np.float32).astype(np.int32)
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_required)
        idx = np.arange(0, n_frames, step, dtype=np.float32).astype(np.int32)

    sampled_frames = np.array(frames)[idx]
    assert len(idx) == n_required
    assert len(sampled_frames) == n_required
    return sampled_frames

def __sample_frames_for_i3d(frames, n_required):
    # i3d model accepts sequence of 8 frames
    n_frames = len(frames)
    segment_length = 8
    n_segments = int(n_required / segment_length)

    assert n_required % segment_length == 0
    assert n_frames > segment_length

    if n_frames < n_required:
        step = (n_frames - segment_length) / float(n_segments)
        idces_start = np.arange(0, n_frames - segment_length, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()

    sampled_frames = np.array(frames)[idx]
    return sampled_frames

def __sample_frames_for_non_local(frames, n_required):
    # i3d model accepts sequence of 8 frames
    n_frames = len(frames)
    segment_length = 128
    n_segments = int(n_required / segment_length)

    assert n_required % segment_length == 0
    assert n_frames > segment_length

    if n_frames < n_required:
        step = (n_frames - segment_length) / float(n_segments)
        idces_start = np.arange(0, n_frames - segment_length, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()

    sampled_frames = np.array(frames)[idx]
    return sampled_frames

# endregion

# region 3.0 Extract Features

def _301_extract_features_i3d():
    n_frames_per_video = 128
    n_frames_per_video = 256
    n_frames_per_video = 512
    n_frames_per_video = 1024

    feature_name = 'mixed_5c'
    annot_actions_path = Pth('Breakfast/annotation/annot_unit_actions.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    video_ids_path = Pth('Breakfast/annotation/video_ids.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames_per_video,))

    video_ids = utils.pkl_load(video_ids_path)
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    features_root_path = Pth('Breakfast/features_i3d_%s_%d_frames', (feature_name, n_frames_per_video))
    if not os.path.exists(features_root_path):
        print ('Sorry, feature path does not exist: %s' % (features_root_path))
        return

    video_frames_dict = {}
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)
    del video_frames_dict_tr
    del video_frames_dict_te

    n_threads = 8
    n_frames_per_segment = 8
    n_segments_per_video = int(n_frames_per_video / n_frames_per_segment)
    n_videos = len(video_ids)

    assert n_frames_per_segment * n_segments_per_video == n_frames_per_video

    # aync reader, and get load images for the first video
    f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[0], n)) for n in video_frames_dict[video_ids[0]]])
    img_reader = AsyncImageReaderBreakfastForI3DKerasModel(n_threads=n_threads)
    img_reader.load_imgs_in_batch(f_pathes)

    # initialize the model
    model = __get_i3d_model_mixed_5c()

    for idx_video, video_id in enumerate(video_ids):

        video_num = idx_video + 1

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            threading._sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print ('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))

        # get the video frames
        video_frames = img_reader.get_images()

        # reshape to get the segments in one dimension
        frames_shape = video_frames.shape
        frames_shape = [n_segments_per_video, n_frames_per_segment] + list(frames_shape[1:])
        video_frames = np.reshape(video_frames, frames_shape)

        # pre-load for the next video
        if video_num < n_videos:
            next_f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[idx_video + 1], n)) for n in video_frames_dict[video_ids[idx_video + 1]]])
            img_reader.load_imgs_in_batch(next_f_pathes)

        # features path
        features_path = '%s/%s.pkl' % (features_root_path, video_id)

        # extract features
        features = model.predict(video_frames, verbose=0)

        # squeeze
        features = np.squeeze(features, axis=1)

        # save features
        utils.pkl_dump(features, features_path)

def _303_extract_features_resnet(idx_start, idx_end, core_id):
    """
    Extract frames from each video. Extract only 1 frame for each spf seconds.
    :param spf: How many seconds for each sampled frames.
    :return:
    """

    __config_session_for_keras(core_id)

    frames_annot_path = Pth('Breakfast/annotation/annot_frames_resnet_64.pkl')
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    video_names_tr = video_frames_dict_tr.keys()
    video_names_te = video_frames_dict_te.keys()

    video_names = np.hstack((video_names_tr, video_names_te))
    video_names = natsort.natsorted(video_names)
    video_names = np.array(video_names)[idx_start:idx_end]
    n_videos_names = len(video_names)
    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)

    feature_name = 'res5c'
    n_frames_per_video = 64
    features_root_path = '/ssd/nhussein/Breakfast/features_resnet_%s_%d_frames' % (feature_name, n_frames_per_video)
    frames_root_path = '/ssd/nhussein/Breakfast/frames'
    if not os.path.exists(features_root_path):
        print os.mkdir(features_root_path)

    batch_size = 80
    bgr_mean = np.array([103.939, 116.779, 123.680])

    # load model
    model = ResNet152(include_top=False, weights='imagenet')
    model.trainable = False
    print (model.summary())

    # loop on videos
    for idx_video, video_id in enumerate(video_names):
        video_num = idx_video + 1
        video_features_path = '%s/%s.pkl' % (features_root_path, video_id)

        # read frames of the video (in batches), and extract features accordingly

        frames_pathes = video_frames_dict[video_id]
        frames_pathes = np.array(['%s/%s/%s' % (frames_root_path, video_id, n) for n in frames_pathes])

        t1 = time.time()

        # read images
        video_imgs = __read_and_preprocess_images(frames_pathes, bgr_mean)

        # extract features
        video_features = model.predict(video_imgs, batch_size)

        # save features
        utils.pkl_dump(video_features, video_features_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, idx_end, video_id, duration))

def __read_and_preprocess_images(img_pathes, bgr_mean):
    n_imgs = len(img_pathes)
    imgs = np.zeros((n_imgs, 224, 224, 3), np.float32)

    for idx, img_path in enumerate(img_pathes):
        # read image
        img = cv2.imread(img_path)
        img = img.astype(np.float32)

        # subtract mean pixel from image
        img[:, :, 0] -= bgr_mean[0]
        img[:, :, 1] -= bgr_mean[1]
        img[:, :, 2] -= bgr_mean[2]

        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        imgs[idx] = img

    return imgs

def __get_i3d_model_mixed_5c():
    NUM_CLASSES = 400
    input_shape = (64, 224, 224, 3)

    # build model for RGB data, and load pretrained weights (trained on imagenet and kinetics dataset)
    i3d_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=input_shape, classes=NUM_CLASSES)

    # set model as non-trainable
    for layer in i3d_model.layers:
        layer.trainable = False
    i3d_model.trainable = False

    return i3d_model

def __get_i3d_model_softmax():
    NUM_CLASSES = 400
    input_shape = (8, 224, 224, 3)

    # build model for RGB data, and load pretrained weights (trained on imagenet and kinetics dataset)
    i3d_model = Inception_Inflated3d(include_top=True, weights='rgb_imagenet_and_kinetics', input_shape=input_shape, classes=NUM_CLASSES)

    # set model as non-trainable
    for layer in i3d_model.layers:
        layer.trainable = False
    i3d_model.trainable = False

    return i3d_model

def __config_session_for_keras(gpu_core_id):
    import keras.backend as K
    import tensorflow as tf

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(gpu_core_id)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

# endregion

# region 4.0 Pickle Features

def _401_pickle_features_i3d_mixed_5c():
    n_frames_per_video = 512
    features_root_path = Pth('Breakfast/features_i3d_mixed_5c_%d_frames', (n_frames_per_video,))
    features_path = Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames.h5', (n_frames_per_video,))
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    n_frames_per_segment = 8
    n_segments = int(n_frames_per_video / n_frames_per_segment)
    assert n_segments * n_frames_per_segment == n_frames_per_video

    f_tr = np.zeros((n_tr, n_segments, 7, 7, 1024), dtype=np.float16)
    f_te = np.zeros((n_te, n_segments, 7, 7, 1024), dtype=np.float16)

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_tr[i])
        f = utils.pkl_load(p)  # (T, 7, 7, 2048)
        f_tr[i] = f

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_te[i])
        f = utils.pkl_load(p)  # (T, 7, 7, 2048)
        f_te[i] = f

    print f_tr.shape
    print f_te.shape

    print(utils.get_size_in_gb(utils.get_array_memory_size(f_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f_tr, f_te), data_names, features_path)

# endregion

# region 5.0 Generate Centroids

def _501_generate_centroids(n_centroids, n_dims):
    c1_path = Pth('Breakfast/features_centroids/features_random_%d_centroids.pkl', (n_centroids,))
    c2_path = Pth('Breakfast/features_centroids/features_sobol_%d_centroids.pkl', (n_centroids,))

    # centroids as random vectors
    c1 = np.random.rand(n_centroids, n_dims)

    # centroids as sobol sequence
    c2 = sobol.sobol_generate(n_dims, n_centroids)
    c2 = np.array(c2)

    # save centroids
    utils.pkl_dump(c1, c1_path)
    utils.pkl_dump(c2, c2_path)

# endregion
