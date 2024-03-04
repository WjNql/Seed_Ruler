# -*- coding: utf-8 -*-
import numpy as np
from skimage.measure import label, regionprops
from interface.IP_Functions import getOptimalSingleGrainArea_1
import cv2


def grainTypeParameter(filePath):
    # 一、Elliptical fitting to separate adherent grains
    global bw_s1, touchedBW, opt_perimeter, one_grain_area

    grain_bw1, labeled, opt_grain_index, numObjects, each_grain_area, one_grain_area, totalNum = getOptimalSingleGrainArea_1.getOptimalSingleGrainArea(filePath)

    image_part = np.uint8(labeled == (opt_grain_index + 1))

    image_part = label(image_part)  # Get the labeled image
    girth = regionprops(image_part)
    girth = np.array([prop.perimeter for prop in girth])
    opt_perimeter = girth[0]  # Optimal perimeter of a single grain

    bw_s1 = grain_bw1.copy()
    touchedBW = grain_bw1.copy()

    for i in range(numObjects):
        if 0.5 * one_grain_area < each_grain_area[i] < 1.4 * one_grain_area:
            touchedBW[labeled == (i + 1)] = 0  # Remove single grain from binary image

    common_part = np.logical_and(bw_s1, touchedBW).astype(np.int64)
    bw_s1 = bw_s1 - common_part

    # touchedBW = imfill(touchedBW, 'holes');
    touchedBW = touchedBW.astype(np.uint8)
    contours, _ = cv2.findContours(touchedBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(touchedBW, [contour], 0, 255, cv2.FILLED)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    touchedBW = cv2.morphologyEx(touchedBW, cv2.MORPH_OPEN, se)

    touchedBW[touchedBW == 255] = 1

    numObjects1, labeled1 = cv2.connectedComponents(touchedBW, connectivity=4)
    numObjects1 = np.max(labeled1)  # Get the maximum number of connected components

    image = touchedBW.copy()  # Contains only adherent grains
    I6 = np.zeros(image.shape)  # Used to accumulate separately segmented grain images
    s_labeled = np.zeros(image.shape)  # Used to label separately segmented grain images
    flag = 1  # Control whether the while loop terminates or not

    return labeled, one_grain_area, totalNum, grain_bw1
