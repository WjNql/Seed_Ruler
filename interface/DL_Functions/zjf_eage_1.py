import cv2
import math
import os
import glob
from pathlib import Path
import numpy as np

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    """Get contours of rectangular objects and their four corner points arranged in the order of top-left -> top-right -> bottom-left -> bottom-right."""
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.medianBlur(imgGray, 7)

    # Convert to binary image
    ret, binary = cv2.threshold(imgBlur, 120, 255, cv2.THRESH_BINARY)
    imgCanny = cv2.Canny(binary, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)

    if showCanny:
        cv2.imshow('Canny', imgThre)
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalCountours

def reorder(myPoints):
    """Reorder the corner points obtained from contours in the order of top-left -> top-right -> bottom-left -> bottom-right."""
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def findDis(pts1, pts2):
    """Calculate the length and width of the rectangular object on A4 paper."""
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5

def count_pixel(image_path):
    image = cv2.imread(image_path)
    save_dir = str(Path('./zjf_eage_images/').absolute())
    save_path = str(Path(save_dir) / Path(image_path).name)
    imgContour, contours = getContours(image, minArea=50000, filter=4, draw=False)
    if len(contours) != 0:
        biggest = contours[0][2]
        newPoints = reorder(biggest)
        newWidth_1 = round((findDis(newPoints[0][0], newPoints[1][0])), 1)
        newHeight_1 = round((findDis(newPoints[0][0], newPoints[2][0])), 1)
        newWidth_2 = round((findDis(newPoints[2][0], newPoints[3][0])), 1)
        newHeight_2 = round((findDis(newPoints[1][0], newPoints[3][0])), 1)

        cv2.line(image, tuple(newPoints[0][0]), tuple(newPoints[1][0]), (0, 255, 0), 2)
        cv2.line(image, tuple(newPoints[0][0]), tuple(newPoints[2][0]), (0, 255, 0), 2)
        cv2.line(image, tuple(newPoints[2][0]), tuple(newPoints[3][0]), (0, 255, 0), 2)
        cv2.line(image, tuple(newPoints[1][0]), tuple(newPoints[3][0]), (0, 255, 0), 2)

        if newWidth_1 < 1000 or newHeight_1 < 1000 or newWidth_2 < 1000 or newHeight_2 < 1000:
            pixel = 0
        else:
            pixel = (210 + 279) * 2 / (newWidth_1 + newHeight_1 + newWidth_2 + newHeight_2)
    else:
        pixel = 0
    if pixel == 0:
        print(image_path)
    return round(pixel, 6)

if __name__ == '__main__':
    source = './inference/CSLL/'
    p = str(Path(source).absolute())
    files = glob.glob(os.path.join(p, '*.*'))
    images = [x for x in files if x.split('.')[-1].lower() in img_formats]
    for i, imgpath in enumerate(images):
        px = count_pixel(imgpath)
        print(px)
