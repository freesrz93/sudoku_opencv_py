# -*- coding: utf-8 -*-
import cv2
import numpy as np
import plotCVImg
from config import *


def correct2(img_original):

    if DEBUG:
        plotCVImg.plotImg(img_original, "original")

    # gray image
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        plotCVImg.plotImg(img_gray, "gray")

    # median Blur
    img_blur = cv2.medianBlur(img_gray, 5)
    if DEBUG:
        plotCVImg.plotImg(img_blur, "median Blur")

    # Gaussian Blur
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    if DEBUG:
        plotCVImg.plotImg(img_blur, "GaussianBlur")

    # adaptive threshold
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if DEBUG:
        plotCVImg.plotImg(img_thresh, "adaptiveThreshold")

    # find the contours RETR_EXTERNAL
    binary, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        plotCVImg.plotImg(img_contours, "contours")

    # 合并轮廓
    merge_list = contours
    contours_merge = np.vstack([merge_list[0], merge_list[1]])
    for i in range(2, len(merge_list)):
        contours_merge = np.vstack([contours_merge, merge_list[i]])

    rect2 = cv2.minAreaRect(contours_merge)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    contours = [box2]
    if DEBUG:
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        plotCVImg.plotImg(img_contours, "contours_merge")

    # find the biggest contours
    size_rectangle_max = 0
    index_biggest = 0
    for i in range(len(contours)):
        size_rectangle = cv2.contourArea(contours[i])
        # store the index of the biggest
        if size_rectangle > size_rectangle_max:
            size_rectangle_max = size_rectangle
            index_biggest = i

    # 多边形拟合
    epsilon = 0.1 * cv2.arcLength(contours[index_biggest], True)
    biggest_rectangle = cv2.approxPolyDP(contours[index_biggest], epsilon, True)

    if DEBUG:
        # copy the original image to show the border
        img_border = img_original.copy()
        # 画出数独方格的边界
        for x in range(len(biggest_rectangle)):
            cv2.line(img_border,
                     (biggest_rectangle[(x % 4)][0][0], biggest_rectangle[(x % 4)][0][1]),
                     (biggest_rectangle[((x + 1) % 4)][0][0], biggest_rectangle[((x + 1) % 4)][0][1]),
                     (255, 0, 0), 2)
        plotCVImg.plotImg(img_border, "border")

    # sort the corners to remap the image
    def sortCornerPoints(rcCorners):
        point = rcCorners.reshape((4, 2))
        mean = rcCorners.sum() / 8
        cornerPoint = np.zeros((4, 2), dtype=np.float32)
        for i in range(len(point)):
            if point[i][0] < mean:
                if point[i][1] < mean:
                    cornerPoint[0] = point[i]
                else:
                    cornerPoint[2] = point[i]
            else:
                if point[i][1] < mean:
                    cornerPoint[1] = point[i]
                else:
                    cornerPoint[3] = point[i]
        return cornerPoint

    # 透视变换
    cornerPoints = sortCornerPoints(biggest_rectangle)
    puzzlePoints = np.float32([[0, 0], [SIZE_PUZZLE, 0], [0, SIZE_PUZZLE], [SIZE_PUZZLE, SIZE_PUZZLE]])
    PerspectiveMatrix = cv2.getPerspectiveTransform(cornerPoints, puzzlePoints)
    img_puzzle = cv2.warpPerspective(img_thresh, PerspectiveMatrix, (SIZE_PUZZLE, SIZE_PUZZLE))
    if DEBUG:
        plotCVImg.plotImg(img_puzzle, "puzzle")

    return img_puzzle
