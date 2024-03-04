# -*- coding: utf-8 -*-
from skimage.measure import label, regionprops
import numpy as np
import cv2
from interface.IP_Functions import grainSegmentation_1

def getOptimalSingleGrainArea(filePath):
    global one_grain_area
    global each_grain_area

    # I. Input Data Preparation

    # 1. [labeled, numObjects] = bwlabel(grain_bw, 4)
    grain_bw = grainSegmentation_1.grainSegmentation(filePath)
    grain_bw = np.uint8(grain_bw)  # Convert image data type to uint8

    numObjects, labeled = cv2.connectedComponents(grain_bw, connectivity=4)

    # Get the maximum number of connected components
    numObjects = np.max(labeled)

    # 2. regionprops and cell2mat corresponding function: each_grain_area results are not quite right
    l = label(labeled)  # Get the labeled image
    each_grain_area = regionprops(l)
    each_grain_area = np.array([prop.area for prop in each_grain_area])

    # 3. sort_each_grain_area and indx
    sort_each_grain_area = np.sort(each_grain_area)
    indx = np.argsort(each_grain_area)
    smooth_sort_each_grain_area = sort_each_grain_area

    # 4. sort_each_grain_area_grad2
    sort_each_grain_area_grad = np.gradient(smooth_sort_each_grain_area, 1)
    sort_each_grain_area_grad2 = np.gradient(sort_each_grain_area_grad, 1)

    # II. Calculate the Optimal Single Grain Area

    numObjects = len(sort_each_grain_area_grad2)
    start0 = 0
    end0 = 0

    for i in range(1, numObjects - 1):
        if sort_each_grain_area_grad2[i - 1] < 0 and sort_each_grain_area_grad2[i + 1] > 0:
            start0 = i
            break

    for i in range(start0, numObjects - 1):
        temp = (sort_each_grain_area[i + 3] - sort_each_grain_area[i]) / sort_each_grain_area[i]
        if temp > 0.2:
            end0 = i
            break

    if end0 == 0:
        end0 = numObjects

    # III. Determine the Optimal Single Grain Area

    min_error = 1000000

    for i in range(start0, end0 + 1):
        total_error = 0
        total_number = 0
        for j in range(start0, end0 + 1):
            temp1 = int(sort_each_grain_area[j] / sort_each_grain_area[i])
            temp3 = sort_each_grain_area[j] / sort_each_grain_area[i]
            if temp1 == 0 or temp1 == 1:
                total_error += abs(1 - temp3)
                total_number += 1

        total_error /= total_number
        if total_error < min_error:
            min_error = total_error
            one_grain_area = sort_each_grain_area[i]
            opt_grain_index = indx[i]

    numObjects = len(each_grain_area)
    threshold2 = 0.5
    threshold1 = 0.5
    contain = np.zeros((numObjects, 2))

    for i in range(numObjects):
        temp1 = int((each_grain_area[i] / one_grain_area))
        temp2 = each_grain_area[i] / one_grain_area - temp1
        contain[i, 0] = i + 1

        if temp1 == 0 and temp2 > threshold2:
            contain[i, 1] = 1
        elif temp1 > 0 and temp2 > threshold1:
            contain[i, 1] = temp1 + 1
        elif temp1 > 0 and temp2 < threshold1:
            contain[i, 1] = temp1

    totalNum = np.sum(contain[:, 1])
    return grain_bw, labeled, opt_grain_index, numObjects, each_grain_area, one_grain_area, totalNum
