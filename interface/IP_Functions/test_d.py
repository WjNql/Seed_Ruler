# -*- coding: utf-8 -*-
import numpy as np
import cv2
from interface.IP_Functions import grainTypeParameter
from skimage import morphology
from scipy import ndimage as ndi
from numba import jit
from skimage.morphology import dilation, disk
from skimage.measure import regionprops
from skimage.measure import label
from PIL import Image
import os




def getRateofseeds(filePath, save_dir):
    # I. Preprocessing
    labeled_image, one_grain_area, totalNum, finalGrainBw_1 = grainTypeParameter.grainTypeParameter(filePath)
    totalNum_1, finalGrainBw_labeled = cv2.connectedComponents(finalGrainBw_1, connectivity=4)

    # 1. Grayscale image
    I2 = cv2.imread(filePath)

    # Convert image from BGR to RGB color channel order
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    I_gray = I2[:, :, 0]
    I_gray1 = I2[:, :, 1]
    size1 = I_gray.shape

    # 2. Threshold segmentation
    white_threshold = 160  # Define the threshold for white, with grayscale greater than 160

    @jit(parallel=True)
    def getWhiteConnected(height, width):
        white_connected = np.zeros((size1[0], size1[1]))
        for i in range(height):
            for j in range(width):
                if (labeled_image[i, j] == 0 and I_gray[i, j] > white_threshold and I_gray1[i, j] > white_threshold):
                    white_connected[i, j] = 1
                else:
                    white_connected[i, j] = 0
        return white_connected

    white_connected = getWhiteConnected(size1[0], size1[1])

    # 3. Remove small and large buds
    min_bud_area = int(one_grain_area / 50)  # Determine the minimum bud area
    max_bud_area = int(1.5 * one_grain_area)  # Determine the maximum bud area
    white_connected = white_connected > 0
    white_connected = morphology.remove_small_objects(white_connected, min_bud_area, connectivity=1)

    def remove_max_objects(ar, max_size, connectivity=1, in_place=False):

        # Raising type error if not int or bool
        if in_place:
            out = ar
        else:
            out = ar.copy()

        if max_size == 0:  # shortcut for efficiency
            return out

        if out.dtype == bool:
            selem = ndi.generate_binary_structure(ar.ndim, connectivity)
            ccs = np.zeros_like(ar, dtype=np.int32)
            ndi.label(ar, selem, output=ccs)
        else:
            ccs = out

        try:
            component_sizes = np.bincount(ccs.ravel())
        except ValueError:
            raise ValueError("Negative value labels are not supported. Try "
                             "relabeling the input with `scipy.ndimage.label` or "
                             "`skimage.morphology.label`.")

        if len(component_sizes) == 2 and out.dtype != bool:
            print("Only one label was provided to `remove_small_objects`. "
                  "Did you mean to use a boolean array?")

        too_max = component_sizes > max_size
        too_max_mask = too_max[ccs]
        out[too_max_mask] = 0

        return out

    white_connected_1 = remove_max_objects(white_connected, max_bud_area)
    white_connected_1 = white_connected_1.astype(np.uint8)
    I2 = I2.astype(np.uint8)

    # II. Counting the number of buds

    # 1. Marking whether white connected regions are connected to grain-connected regions
    white_numObjects, white_labeled = cv2.connectedComponents(white_connected_1, connectivity=4)
    white_numObjects = np.max(white_labeled)  # Get the maximum number of connected components

    isWhiteConnectGrain = np.zeros(
        white_numObjects)  # Mark whether white connected regions are connected to grain-connected regions
    for i in range(1, size1[0] - 1):
        for j in range(1, size1[1] - 1):
            temp1 = white_labeled[i, j]
            if temp1 == 0:
                continue
            if isWhiteConnectGrain[temp1 - 1] == 1:
                continue
            if temp1 > 0 and (
                    labeled_image[i - 1, j] > 0 or labeled_image[i + 1, j] > 0 or labeled_image[i, j - 1] > 0 or
                    labeled_image[i, j + 1] > 0):
                isWhiteConnectGrain[temp1 - 1] = 1

    # 2. White regions connected to grains
    new_white_connected = np.zeros((size1[0], size1[1]))  # White regions connected to grains
    for i in range(1, size1[0] - 1):
        for j in range(1, size1[1] - 1):
            temp1 = white_labeled[i, j]
            if temp1 == 0:
                continue
            if isWhiteConnectGrain[temp1 - 1] == 1:
                new_white_connected[i, j] = 1

    new_white_labeled, _ = label(new_white_connected, connectivity=2, return_num=True)
    new_white_numObjects = np.max(new_white_labeled)

    new_white_connected_perimeter = regionprops(new_white_labeled)
    new_white_connected_perimeter = np.array([prop.perimeter for prop in new_white_connected_perimeter])

    mark_touch_area = np.zeros(new_white_numObjects)
    touchPixelsNum = np.zeros(new_white_numObjects)
    isTouchPixels = np.zeros((size1[0], size1[1]))

    for i in range(1, size1[0] - 1):
        for j in range(1, size1[1] - 1):
            temp1 = new_white_labeled[i, j]

            if temp1 == 0:
                continue

            if temp1 > 0 and (
                    labeled_image[i - 1, j] > 0 or labeled_image[i + 1, j] > 0 or labeled_image[i, j - 1] > 0 or
                    labeled_image[i, j + 1] > 0):
                if labeled_image[i - 1, j] > 0:
                    mark_touch_area[temp1 - 1] = labeled_image[i - 1, j]
                    touchPixelsNum[temp1 - 1] += 1
                    isTouchPixels[i, j] = temp1
                    continue
                elif labeled_image[i + 1, j] > 0:
                    mark_touch_area[temp1 - 1] = labeled_image[i + 1, j]
                    touchPixelsNum[temp1 - 1] += 1
                    isTouchPixels[i, j] = temp1
                    continue
                elif labeled_image[i, j - 1] > 0:
                    mark_touch_area[temp1 - 1] = labeled_image[i, j - 1]
                    touchPixelsNum[temp1 - 1] += 1
                    isTouchPixels[i, j] = temp1
                    continue
                else:
                    mark_touch_area[temp1 - 1] = labeled_image[i, j + 1]
                    touchPixelsNum[temp1 - 1] += 1
                    isTouchPixels[i, j] = temp1

    div1 = np.zeros(new_white_numObjects)
    mark_whiteRegion_isValid = np.zeros(new_white_numObjects)
    for i in range(new_white_numObjects):
        div1[i] = touchPixelsNum[i] / new_white_connected_perimeter[i]
        if div1[i] < 0.4:
            mark_whiteRegion_isValid[i] = 1

    new_white_connected3 = np.zeros((size1[0], size1[1]))
    for i in range(size1[0]):
        for j in range(size1[1]):
            l = new_white_labeled[i, j]
            if l == 0:
                continue
            if mark_whiteRegion_isValid[l - 1] == 0:
                new_white_connected3[i, j] = 0
            if mark_whiteRegion_isValid[l - 1] == 1:
                new_white_connected3[i, j] = 1

    new_white_labeled3, _ = label(new_white_connected3, connectivity=2, return_num=True)
    new_white_numObjects3 = np.max(new_white_labeled3)

    # 3. Bud count calculation
    threshold2 = 0.5
    threshold1 = 0.5

    def cnt_process(grain_area):
        contain = 0
        temp1 = int((grain_area / one_grain_area))  # Get the integer part
        temp2 = grain_area / one_grain_area - temp1  # Get the decimal part

        if temp1 == 0 and temp2 > threshold2:
            contain = 1
        elif temp1 > 0 and temp2 > threshold1:
            contain = temp1 + 1
        elif temp1 > 0 and temp2 < threshold1:
            contain = temp1
        return contain

    def count_connected_grains_1(labeled_1, labeled_2):
        # Connected components analysis

        labeled_grain = label(labeled_1)
        labeled_bud = label(labeled_2)

        # Get connected components properties for grains and buds
        grain_props = regionprops(labeled_grain)
        bud_props = regionprops(labeled_bud)

        # Initialize grain count
        bud_count_1 = 0
        # List to track whether buds are assigned to grains
        bud_assigned = [False] * len(bud_props)
        # Iterate through each grain
        for grain in grain_props:
            # Create a circular structuring element with a radius of 3
            selem = disk(3)
            # Perform dilation operation
            dilated_grain = dilation(labeled_grain == grain.label, selem)

            # Initialize flag for whether connected to buds
            connected_buds_1 = 0
            # Iterate through each bud
            for i, bud in enumerate(bud_props):
                if bud_assigned[i]:
                    continue
                selem = disk(1)
                dilated_buds = dilation(labeled_bud == bud.label, selem)

                # Dilate grain-bud connected region
                # dilated_buds = binary_dilation(labeled_bud == bud.label)
                # Check if connected regions overlap
                intersection = np.logical_and(dilated_grain, dilated_buds)
                if np.any(intersection):
                    bud_assigned[i] = True
                    connected_buds_1 += 1

            cnt = cnt_process(grain.area)
            if connected_buds_1:
                if connected_buds_1 > cnt:
                    connected_buds_1 = cnt
            # If connected to buds, increment grain count
            bud_count_1 += connected_buds_1
        return bud_count_1

    bud_count_1 = count_connected_grains_1(finalGrainBw_1, new_white_connected3)

    # 4. Displaying buds
    height, width, channels = I2.shape

    # Create a three-channel matrix with zeros of the same size as I2
    I_tmp = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(size1[0]):
        for j in range(size1[1]):
            if finalGrainBw_1[i, j] == 1:
                I_tmp[i, j, 0] = 255
                I_tmp[i, j, 1] = 255
                I_tmp[i, j, 2] = 0
            if new_white_connected3[i, j] == 1:
                I_tmp[i, j, 0] = 255
                I_tmp[i, j, 1] = 0
                I_tmp[i, j, 2] = 255

    # Create a PIL image object
    pil_img = Image.fromarray(I_tmp)

    # Save the image in PNG format
    filename = filePath.split('/')[-1].split('.')[0]
    new_file_path = os.path.join(save_dir, filename + '.png')
    pil_img.save(new_file_path)

    I_tmp_0 = np.zeros((height, width, 3), dtype=np.uint8)

    # Set the red channel to 0
    I_tmp_0[:, :, 0] = 0
    # Set the green channel to 0
    I_tmp_0[:, :, 1] = 0
    # Set the blue channel to 0
    I_tmp_0[:, :, 2] = 0
    for i in range(size1[0]):
        for j in range(size1[1]):
            if finalGrainBw_1[i, j] == 1:
                I_tmp_0[i, j, 0] = 255
                I_tmp_0[i, j, 1] = 255
                I_tmp_0[i, j, 2] = 0
            if white_connected[i, j] == 1:
                I_tmp_0[i, j, 0] = 255
                I_tmp_0[i, j, 1] = 0
                I_tmp_0[i, j, 2] = 255

    I_tmp_1 = np.zeros((height, width, 3), dtype=np.uint8)

    # Set all pixels in the matrix to black
    I_tmp_1[:, :, 0] = 0
    # Set the green channel to 0
    I_tmp_1[:, :, 1] = 0
    # Set the blue channel to 0
    I_tmp_1[:, :, 2] = 0
    for i in range(size1[0]):
        for j in range(size1[1]):
            if finalGrainBw_1[i, j] == 1:
                I_tmp_1[i, j, 0] = 255
                I_tmp_1[i, j, 1] = 255
                I_tmp_1[i, j, 2] = 0
            if white_connected_1[i, j] == 1:
                I_tmp_1[i, j, 0] = 255
                I_tmp_1[i, j, 1] = 0
                I_tmp_1[i, j, 2] = 255

        # 5. Germination rate result
    return {"Name": filename, "Total Grains": totalNum, "Germinated Grains": bud_count_1,
            "Germination Rate": bud_count_1 / totalNum}


