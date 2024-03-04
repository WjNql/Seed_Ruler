# -*- coding: utf-8 -*-
import numpy as np
import cv2
from scipy.ndimage import label, binary_erosion
from skimage.morphology import binary_erosion, binary_dilation
import scipy.ndimage as ndimage
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def getRegionConcavePoints(image_part_i):
    n, l = cv2.connectedComponents(image_part_i.astype(np.uint8), connectivity=8)

    boundaryBw = np.logical_and(image_part_i, np.logical_not(binary_erosion(image_part_i)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(boundaryBw.astype(np.uint8), connectivity=8)

    # Set area threshold
    area_threshold = 5

    # Iterate over connected components' statistics
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= area_threshold:
            labels[labels == label] = 0

    boundaryBw = labels.astype(bool)

    b_labeled, b_numObjects = ndimage.label(boundaryBw, structure=np.ones((3, 3), dtype=np.int))

    x = []
    y = []

    for k in range(1, b_numObjects + 1):
        tmp = (b_labeled == k)
        tmp = tmp.astype(np.uint8)
        contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bound = contours[0]
        length = len(bound)

        nonzero_indices = np.transpose(np.nonzero(tmp))
        first_nonisolated_index = nonzero_indices[0]

        bound_start_index = np.argmax(np.all(bound == first_nonisolated_index, axis=2))
        bound = np.roll(bound, -bound_start_index, axis=0)

        x_min_index = np.argmin(bound[:, 0, 0])
        bound = np.roll(bound, -x_min_index, axis=0)

        bound = np.flip(bound, axis=1)
        bound = np.flip(bound.transpose(1, 0, 2), axis=0)

        b = bound.reshape(-1, 2)
        x = b[:, 0]
        y = b[:, 1]
        x = x[::-1]
        y = y[::-1]

        angle_threshold = 220

        def is_white_pixel(point, image):
            x, y = point
            pixel_value = image[y, x]
            if pixel_value > 0:
                return True
            else:
                return False

        step = 7
        x1 = x.reshape(-1, 1)
        y1 = y.reshape(-1, 1)
        x1 = x1.flatten()
        y1 = y1.flatten()
        len_ = len(y1) + 2 * step
        new_x = np.zeros(len_)
        new_x[step:len_ - step] = x1
        new_x[0:step] = x1[-step:]
        new_x[len_ - step:len_] = x1[0:step]
        new_y = np.zeros(len_)
        new_y[step:len_ - step] = y1
        new_y[0:step] = y1[-step:]
        new_y[len_ - step:len_] = y1[0:step]
        arc_length = np.zeros(len(y1))

        for j in range(len(y1)):
            radius = 8.6
            later_vec = np.vstack((new_x[j:j + step + 1], new_y[j:j + step + 1]))
            pre_vec = np.vstack((new_x[j + step:j + 2 * step + 1], new_y[j + step:j + 2 * step + 1]))
            later_unique_x, later_unique_idx = np.unique(later_vec[0], return_index=True)
            later_unique_y = later_vec[1][later_unique_idx]
            pre_unique_x, pre_unique_idx = np.unique(pre_vec[0], return_index=True)
            pre_unique_y = pre_vec[1][pre_unique_idx]

            later_interp_func = interp1d(later_unique_x, later_unique_y, kind='linear')
            later_fited_y = later_interp_func(later_vec[0])

            pre_interp_func = interp1d(pre_unique_x, pre_unique_y, kind='linear')
            pre_fited_y = pre_interp_func(pre_vec[0])

            later_coeff = np.polyfit(later_vec[0], later_fited_y, deg=1)
            pre_coeff = np.polyfit(pre_vec[0], pre_fited_y, deg=1)

            a1 = later_coeff[0]
            a2 = pre_coeff[0]

            if np.max(later_vec[0, :]) - np.min(later_vec[0, :]) <= 2:
                a1 = 20000
            if np.max(pre_vec[0, :]) - np.min(pre_vec[0, :]) <= 2:
                a2 = 18000

            l_vec = np.array([later_vec[0, -4] - later_vec[0, -1], later_vec[1, -4] - later_vec[1, -1], 0])
            p_vec = np.array([pre_vec[0, -4] - pre_vec[0, 0], pre_vec[1, -4] - pre_vec[1, 0], 0])
            B_Point_x1 = -np.sqrt((radius) ** 2 / (1 + a1 ** 2))
            B_Point_y1 = a1 * B_Point_x1
            B_Point_x2 = np.sqrt((radius) ** 2 / (1 + a1 ** 2))
            B_Point_y2 = a1 * B_Point_x2
            d1 = (B_Point_x1 - l_vec[0]) ** 2 + (B_Point_y1 - l_vec[1]) ** 2
            d2 = (B_Point_x2 - l_vec[0]) ** 2 + (B_Point_y2 - l_vec[1]) ** 2
            if d1 < d2:
                B_Point_x = B_Point_x1
                B_Point_y = B_Point_y1
            else:
                B_Point_x = B_Point_x2
                B_Point_y = B_Point_y2
            A_Point_x1 = -np.sqrt((radius) ** 2 / (1 + a2 ** 2))
            A_Point_y1 = a2 * A_Point_x1
            A_Point_x2 = np.sqrt((radius) ** 2 / (1 + a2 ** 2))
            A_Point_y2 = a2 * A_Point_x2
            d3 = (A_Point_x1 - p_vec[0]) ** 2 + (A_Point_y1 - p_vec[1]) ** 2
            d4 = (A_Point_x2 - p_vec[0]) ** 2 + (A_Point_y2 - p_vec[1]) ** 2
            if d3 < d4:
                A_Point_x = A_Point_x1
                A_Point_y = A_Point_y1
            else:
                A_Point_x = A_Point_x2
                A_Point_y = A_Point_y2

            theta = np.arange(1, 361)
            circle_x = radius * np.cos(np.deg2rad(theta))
            circle_y = -radius * np.sin(np.deg2rad(theta))
            circle = np.column_stack((circle_x, circle_y))

            tmp = np.array([[A_Point_x, A_Point_y], [B_Point_x, B_Point_y]])
            tree = cKDTree(circle)
            dist, index = tree.query(tmp, k=1)
            index = index.astype(int)
            r_x = circle[index, 0]
            r_y = circle[index, 1]
            A_x = r_x[0]
            A_y = r_y[0]
            B_x = r_x[1]
            B_y = r_y[1]
            A_index = index[0]
            B_index = index[1]

            if A_index < B_index:
                fore_arc_length = A_index + (360 - B_index)
                length1 = B_index - A_index
                length2 = 360 - length1
                n2 = 0
                n3 = 0
                for ind in range(A_index, B_index + 1):
                    if l[int(round(circle_y[ind] + later_vec[1, -1])),
                         int(round(circle_x[ind] + later_vec[0, -1]))] > 0:
                        n2 += 1

                for ind in range(0, A_index + 1):
                    if l[int(round(circle_y[ind] + later_vec[1, -1])),
                         int(round(circle_x[ind] + later_vec[0, -1]))] > 0:
                        n3 += 1

                for ind in range(B_index, 360):
                    if l[int(round(circle_y[ind] + later_vec[1, -1])),
                         int(round(circle_x[ind] + later_vec[0, -1]))] > 0:
                        n3 += 1

                if n2 < n3:
                    fore_arc_length = n3
                else:
                    fore_arc_length = n2

            else:
                fore_arc_length = A_index - B_index
                n2 = 0
                for ind in range(B_index, A_index + 1):
                    if l[int(round(circle_y[ind] + later_vec[1, -1])),
                         int(round(circle_x[ind] + later_vec[0, -1]))] > 0:
                        n2 += 1

                n3 = 0
                for ind in range(0, B_index + 1):
                    if l[int(round(circle_y[ind] + later_vec[1, -1])),
                         int(round(circle_x[ind] + later_vec[0, -1]))] > 0:
                        n3 += 1

                for ind in range(A_index, 360):
                    if l[int(round(circle_y[ind] + later_vec[1, -1])),
                         int(round(circle_x[ind] + later_vec[0, -1]))] > 0:
                        n3 += 1

                if n2 < n3:
                    fore_arc_length = n3
                else:
                    fore_arc_length = n2

            if A_index == B_index:
                fore_arc_length = 350

            arc_length[j] = fore_arc_length

        arc_length_abs = np.abs(arc_length)

        local_maxima = argrelextrema(arc_length_abs, np.greater, order=8)[0]
        TF = np.zeros_like(arc_length, dtype=bool)
        TF[local_maxima] = True

        num_1 = np.sum(TF)
        print("TF", num_1)

        for t in range(len(y1)):
            if arc_length[t] < 220:
                TF[t] = False

        len_ = len(y1)
        num_2 = np.sum(TF)
        print("TF", num_2)
    return TF, len_, x, y
