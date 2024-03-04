# -*- coding: utf-8 -*-
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import disk, binary_opening
from interface.IP_Functions import segmentsFitElip, regionSegments, getRegionConcavePoints
import cv2
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt


def oneSegProcess(image, one_grain_area, opt_perimeter, s_labeled):
    # Elliptical fitting to separate adherent grains
    # s_labeled is the labeled image of single grains obtained in the previous segmentation round
    global numOfConcave, c_row, c_colum
    image_copy = np.copy(image)
    image = image.astype(np.uint8)
    num_labels, labeled_image = cv2.connectedComponents(image, connectivity=4)
    num_labels = np.max(labeled_image)  # Get the maximum number of connected components
    I4 = np.zeros_like(image)  # Store the result after elliptical fitting segmentation
    I5 = np.zeros_like(image)  # Contains only separately segmented grains using elliptical fitting
    for i in range(1, num_labels + 1):
        image_part_i = (labeled_image == i)
        TF, length, x, y = getRegionConcavePoints.getRegionConcavePoints(image_part_i)
        numOfConcave = int(np.sum(TF))
        ind = np.where(TF)[0]
        for j in range(numOfConcave):
            index = ind[j]
            c_row = y[index]  # Current row coordinate of the concave point
            c_colum = x[index]  # Current column coordinate of the concave point
            if (
                s_labeled[c_row - 3, c_colum]
                > 0
                or s_labeled[c_row + 3, c_colum]
                > 0
                or s_labeled[c_row, c_colum - 3]
                > 0
                or s_labeled[c_row, c_colum + 3]
                > 0
            ):
                TF[index] = 0
        numOfConcave = int(np.sum(TF))
        if numOfConcave > 1:
            contour_segments = regionSegments.regionSegments(TF, x, y)
            newImage_part_i = segmentsFitElip.segmentsFitElip(
                image_part_i, contour_segments, one_grain_area, opt_perimeter
            )
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                newImage_part_i.astype(np.uint8), 4
            )
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < int(np.floor(one_grain_area / 5)):
                    labels[labels == i] = 0
            newImage_part_i = labels.astype(bool)
            newImage_part_i = newImage_part_i.astype(np.uint8)
            I4 = np.logical_or(I4, newImage_part_i)
        else:
            I4 = np.logical_or(I4, image_part_i)
    I4 = cv2.UMat(I4.astype(np.uint8))
    num_objects, labeled_I4 = cv2.connectedComponents(I4, connectivity=4)
    num_objects = np.max(labeled_I4)  # Get the maximum number of connected components
    labeled_I4 = labeled_I4.get()
    area_tmp = regionprops(labeled_I4)
    tmp_areas = [prop.area for prop in area_tmp]
    for k in range(len(tmp_areas)):
        if tmp_areas[k] < one_grain_area * 1.3 and tmp_areas[k] > one_grain_area * 0.5:
            s_grain = (labeled_I4 == k + 1)  # A single grain in connected region i
            I5 = np.logical_or(I5, s_grain)
    image = cv2.UMat.get(I4) - I5.astype(np.uint8)  # Adherent grain image
    se = disk(3)
    image = binary_opening(image, selem=se)  # Morphological opening operation to remove spiky edges
    if np.array_equal(image_copy, image):
        flag = 0  # Control whether the while loop terminates
        I5 = image
    else:
        flag = 1  # Control whether the while loop terminates
    # Calculate the minimum area threshold
    min_area_threshold = one_grain_area // 5
    # Remove connected regions smaller than the minimum area threshold
    I5 = remove_small_objects(I5, min_size=min_area_threshold, connectivity=1)
    I5 = I5.astype(np.uint8)
    num_objects, labeled_I5 = cv2.connectedComponents(I5, connectivity=4)
    return image, I5, flag, labeled_I5
