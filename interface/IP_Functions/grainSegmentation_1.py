# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import correlate


def grainSegmentation(f_path):
    global grain_bw

    # 一、Segmentation of grains based on k-means clustering algorithm

    # Read the image, supporting common formats like bmp, jpg, png, tiff, etc.
    I = cv2.imread(f_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    I = np.double(I)
    I_gray = I[:, :, 0]
    size1 = I.shape
    I_column = np.reshape(I, (size1[0] * size1[1], 3))

    thr1 = 60
    thr2 = 60
    I1 = I_column[:, 0] - I_column[:, 2]
    I1[I1 >= thr1] = 30
    I2 = I_column[:, 1] - I_column[:, 2]
    I2[I2 >= thr2] = 30
    I4 = np.sum(I_column, axis=1)
    I1 = np.column_stack((I1, I2))
    np.set_printoptions(threshold=np.inf)

    matrix1 = np.array([[40, 40], [0, 0]])
    kmeans = KMeans(n_clusters=2, init=matrix1)
    kmeans.fit(I1)
    idx = kmeans.labels_
    idx = idx.astype(int) - 1
    temp1 = np.nanmean(I4[idx == 0])
    temp2 = np.mean(I4[idx == 1])
    if temp1 > temp2:
        idx = ~idx
    idx_reshape = np.reshape(idx, (size1[0], size1[1]))
    idx_reshape = np.uint8(idx_reshape)

    # 二、Fine segmentation of grains based on Gaussian filtering

    # 1. Remove smaller connected components
    I1 = idx_reshape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(I1.astype(np.uint8), 4)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            labels[labels == i] = 0

    I1 = labels.astype(bool)
    I1 = I1.astype(np.uint8)

    # 2. Fill holes in contours
    contours, _ = cv2.findContours(I1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(I1, [contour], 0, 255, cv2.FILLED)

    I1 = I1.astype(np.uint8)

    # 3. Label connected components
    numObjects, labeled = cv2.connectedComponents(I1, connectivity=4)
    numObjects = np.max(labeled)

    # 4. Shrink each connected region of grains
    each_grain_area_histogram = np.zeros((numObjects, 256))
    for i in range(size1[0]):
        for j in range(size1[1]):
            temp = labeled[i, j]
            if temp != 0:
                temp1 = I_gray[i, j]
                each_grain_area_histogram[temp - 1, int(temp1)] = each_grain_area_histogram[temp - 1, int(temp1)] + 1

    each_grain_area_histogram_inflection_position = np.zeros((numObjects, 1))

    x = np.arange(1, 257)
    d = 5
    initial1 = 150

    # Smoothing histograms with Gaussian filtering and polynomial fitting
    for i in range(numObjects):
        h = np.multiply(cv2.getGaussianKernel(1, 7.5), (cv2.getGaussianKernel(100, 7.5)).T).flatten()
        arr_2d = np.zeros((100, 100))
        arr_2d[:h.size] = h.reshape((h.size, 1))
        each_grain_area_histogram[i, :] = correlate(each_grain_area_histogram[i, :].astype(np.float64), h, mode='reflect')

        y = each_grain_area_histogram[i, :].astype(np.double)
        r = np.polyfit(x, y, d)

        yvals = np.polyval(r, range(initial1, 256))
        indx1 = np.argmax(yvals)
        indx1 = indx1 + initial1 - 2
        for k in reversed(list(range(2, indx1 + 1))):
            temp1 = r[0] * k**5 + r[1] * k**4 + r[2] * k**3 + r[3] * k**2 + r[4] * k + r[5]
            temp2 = r[0] * (k - 1)**5 + r[1] * (k - 1)**4 + r[2] * (k - 1)**3 + r[3] * (k - 1)**2 + r[4] * (k - 1) + r[5]
            temp3 = r[0] * (k + 1)**5 + r[1] * (k + 1)**4 + r[2] * (k + 1)**3 + r[3] * (k + 1)**2 + r[4] * (k + 1) + r[5]
            if temp1 <= temp2 and temp1 <= temp3:
                break

        each_grain_area_histogram_inflection_position[i, 0] = k

    # 5. Remove shadow parts
    for i in range(size1[0]):
        for j in range(size1[1]):
            temp = labeled[i, j]
            if temp != 0:
                if each_grain_area_histogram_inflection_position[temp - 1, 0] != 0 \
                        and I_gray[i, j] < each_grain_area_histogram_inflection_position[temp - 1, 0]:
                    I1[i, j] = 0
                    labeled[i, j] = 0

    # 6. Remove smaller connected components again
    num, labels, stats, _ = cv2.connectedComponentsWithStats(I1.astype(np.uint8), 4)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            labels[labels == i] = 0

    grain_bw = labels.astype(bool)

    return grain_bw
