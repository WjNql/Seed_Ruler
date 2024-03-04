# -*- coding: utf-8 -*-
import numpy as np
import cv2

def segmentsFitElip(image_part_i, contour_segments, one_grain_area, opt_perimeter):
    len_ = len(contour_segments)
    print(len_)
    tmp = np.zeros(len_)
    for kk in range(len_):
        data = contour_segments[kk]

        x = data[:, 0]
        y = data[:, 1]
        num = len(x)
        print("num", num)
        if (np.max(x) - np.min(x)) < 5 or (np.max(y) - np.min(y)) < 5:
            continue
        tmp[kk-1] = num
        k1 = (y[0] - y[round(0.8*num)]) / (x[0] - x[round(0.8*num)])
        k2 = (y[round(num/2)] - y[-1]) / (x[round(num/2)] - x[-1])
        if np.abs(k1 - k2) < 0.3:
            continue
        p_thr = opt_perimeter * 0.5

        if num > p_thr:
            # Scatter plot
            image_part_i = image_part_i.astype(np.uint8)
            # Method One
            ellipse = cv2.fitEllipse(data)
            cv2.ellipse(image_part_i, ellipse, (0, 255, 0), 2)

    new_image_part_i = image_part_i
    return new_image_part_i
