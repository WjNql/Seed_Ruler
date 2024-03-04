# -*- coding: utf-8 -*-
import numpy as np


def regionSegments(TF, x, y):
    numOfConcave = int(np.sum(TF))
    tmpCP_sets = np.zeros((numOfConcave, 2))
    f = np.where(TF == 1)[0]

    first_ind = f[0]
    last_ind = f[-1]

    tmpCP_sets[:, 0] = x[f]
    tmpCP_sets[:, 1] = y[f]

    # Extract values from x and y corresponding to the indices
    tmpCP_sets = np.column_stack((x[f], y[f]))
    ind = len(f)
    contour_segments = [None] * ind
    ind2 = 1
    color = ['r', 'g', 'm', 'r', 'g', 'm', 'r', 'g']
    fz = 3

    if ind == 1:
        contour_segments[0] = np.column_stack((x, y))
    else:
        n = 0
        while ind2 < ind:
            t1 = f[ind2 - 1]
            t2 = f[ind2]
            ttt = np.column_stack((x[t1 - 1:t2], y[t1 - 1:t2]))
            n += 1
            if ind2 % 2 == 0:
                c = color[0]
            else:
                c = color[1]
            contour_segments[ind2 - 1] = ttt
            ind2 += 1
        ttt2 = np.vstack(
            (np.column_stack((x[last_ind - 1:], y[last_ind - 1:])), np.column_stack((x[:first_ind], y[:first_ind]))))

        contour_segments[ind - 1] = ttt2

    contour_segments = np.array(contour_segments, dtype=object)
    return contour_segments
