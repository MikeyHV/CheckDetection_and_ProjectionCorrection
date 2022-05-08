import argparse
import os
import pip

import proj3Helper

import pytesseract
import argparse
import imutils
import cv2
import cv2 as cv
import re
import numpy as np
import math

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import imutils
# import some common libraries
import numpy as np
import os, json, random
import cv2 as cv

#pip install pyyaml==5.1

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
#print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html

def getAppxCorner(img_dilation):
    ret, thresh = cv2.threshold(img_dilation, 127, 255, 0)
    cnt, hier = cv2.findContours(thresh, 0, 3)
    cnt = cnt[0]
    boo = False
    epsilon = 0.08 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, False)

    contours = sorted([approx], key=cv2.contourArea, reverse=True)
    # find the perimeter of the first closed contour
    perim = cv2.arcLength(contours[0], True)
    # setting the precision
    epsilon = 0.02 * perim
    # approximating the contour with a polygon
    approxCorners = cv2.approxPolyDP(contours[0], epsilon, True)
    # check how many vertices has the approximate polygon
    approxCornersNumber = len(approxCorners)
    if approxCornersNumber == 4:
        pass
    else:
        while len(approxCorners) != 4:  # removes corners until have a trapezoid/rectangle
            dists = []
            indexI = 0
            for i in approxCorners:
                p1 = i[0]
                p1x = p1[0]
                p1y = p1[1]
                indexJ = 0
                for j in approxCorners:
                    p2 = j[0]
                    p2x = p2[0]
                    p2y = p2[1]
                    if p1[0] == p2[0] and p2[1] == p1[1]:
                        pass
                    else:
                        dists.append((math.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2), indexI, indexJ))
                    indexJ += 1
                indexI += 1
            d = sorted(dists, key=lambda x: x[0])
            shortestPair = d[0]
            if approxCorners[shortestPair[1]][0][0] > approxCorners[shortestPair[2]][0][0]:
                approxCorners = np.delete(approxCorners, shortestPair[2], axis=0)
            else:
                approxCorners = np.delete(approxCorners, shortestPair[1], axis=0)
    return approxCorners


def getCheck(image,ogog, templatePath):
    img, out, classes, isABookMask, isABookAndPersonMask, stillPeopleInThere, mask, reshapedImage = proj3Helper.getDetectronOutput(
        ogog)
    if not isABookMask and not isABookAndPersonMask:
        templatedImage, templateMask = proj3Helper.getTemplate(templatePath, image)
        img, out, classes, isABookMask, isABookAndPersonMask, stillPeopleInThere, mask, image = proj3Helper.getDetectronOutput(
            templatedImage, iterations=1)
        if mask is None:
            templatedImage2, templateMask = proj3Helper.getTemplate(templatePath, img, scalingFactor=0.95)
            corners = proj3Helper.approximateCorners(templatedImage2, templateMask)
            dst = proj3Helper.fitScreenToRect(reshapedImage, corners)
        elif stillPeopleInThere:
            templatedImage2, templateMask = proj3Helper.getTemplate(templatePath, img, scalingFactor=0.5)
            corners = proj3Helper.approximateCorners(templatedImage2, templateMask)
            dst = proj3Helper.fitScreenToRect(reshapedImage, corners)
        else:
            corners = proj3Helper.approximateCorners(img, mask)
            dst = proj3Helper.fitScreenToRect(reshapedImage, corners)
    elif isABookAndPersonMask:
        # print("HI")
        corners = proj3Helper.approximateCorners(reshapedImage, mask)
        dst = proj3Helper.fitScreenToRect(reshapedImage, corners)
    else:
        edged = proj3Helper.getCanPipe(reshapedImage, eqHist=False)
        rect = proj3Helper.getRect(edged)
        dst = proj3Helper.fitScreenToRect(reshapedImage, rect)
        finImg, smallBox = proj3Helper.correctPerspectiveAndOrientationGivenCheck(dst)
        dst = finImg
        # pltshow(finImg)
    return dst


def process_img(img_path, templatePath):
    frame_orig = cv2.imread(img_path)
    original = frame_orig.copy()
    # templatePath = "samples/blankcheck2.jpg"

    fin = getCheck(frame_orig, original, templatePath)

    frame_result = cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)
    ### Replace the code above.
    cv2.imshow("Original", frame_orig)
    cv2.imshow("Result", frame_result)
    cv2.waitKey(0)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Check prepartion project")
    parser.add_argument('--input_folder', type=str, default='samples', help='check images folder')
    parser.add_argument("--path_to_template", type = str, default="blankcheck2.jpg")
    
    args = parser.parse_args()
    input_folder = args.input_folder
    templatePath = args.path_to_template
   
    for check_img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, check_img)
        if img_path.lower().endswith(('.png','.jpg','.jpeg', '.bmp', '.gif', '.tiff')):
            process_img(img_path, templatePath)
            