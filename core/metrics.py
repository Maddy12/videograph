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
from sklearn.metrics import average_precision_score

def mean_avg_precision_charades(y_true, y_pred):
    """ Returns mAP """
    m_aps = []
    n_classes = y_pred.shape[1]
    for oc_i in range(n_classes):
        pred_row = y_pred[:, oc_i]
        sorted_idxs = np.argsort(-pred_row)
        true_row = y_true[:, oc_i]
        tp = true_row[sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(y_pred.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    return m_ap

def mean_avg_precision_sklearn(y_true, y_pred):
    # """ Returns mAP """
    n_classes = y_true.shape[1]
    map = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]
    map = np.nan_to_num(map)
    map = np.mean(map)
    return map
