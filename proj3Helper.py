import argparse
import os
import pip


import pytesseract
import argparse
import imutils
import cv2
import cv2 as cv
import re
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import imutils
# import some common libraries
import numpy as np
import os, json, random
import cv2 as cv
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def pltshow(imgzz):
    plt.imshow(cv2.cvtColor(imgzz, cv2.COLOR_BGR2RGB))


def eqhist(img):
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def getCanPipe(image, eqHist=False, blur=True, ):
    if eqHist:
        gray = eqhist(image)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur:
        blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    else:
        blurred = gray
    # 75, 200
    edged = cv2.Canny(blurred, 100, 300)
    return edged


def sortCorners(corners):
    c = []
    idx = 0
    for i in corners.tolist():
        c.append(i[0])
        idx += 1
    c = sorted(c, key=lambda x: x[1])
    topones = [c[0], c[1]]
    bottomones = [c[2], c[3]]
    topones = sorted(topones, key=lambda x: x[0])
    topleft = topones[0]
    topright = topones[1]

    botones = sorted(bottomones, key=lambda x: x[0])
    bottomleft = botones[0]
    bottomright = botones[1]
    cornersz = [topleft, topright, bottomleft, bottomright]
    return cornersz


def getRect(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    receiptCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receiptCnt = approx
            # print(receiptCnt)
            break
    if receiptCnt is None:
        raise Exception(("No outline, oof"))
    return receiptCnt


def drawRect(image, rect):
    output = image.copy()
    cv2.drawContours(output, [rect], -1, (0, 255, 0), 2)
    pltshow(output)


def histeq(rgb_img):
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


def subtractMaskFromImg(img, mask1):
    mask2 = cv2.bitwise_not(mask1)
    return cv2.bitwise_and(img, img, mask=mask2)


def getMaskFromImg(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def fitScreenToRect(image, corners):
    rows,cols,ch = image.shape
    H, W = image.shape[:2]
    pts1 = np.float32(sortCorners(corners))
    pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])
    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(image,M,(W,H))
    # dst = cv2.warpPerspective(image,M,(W + (W * 0.5),H))
    return dst

def correctPerspectiveAndOrientationGivenCheck(dst):
    smallCorners = getRect(getCanPipe(dst, eqHist=True))
    closestToBottomRight = sortCorners(smallCorners)[3]  # this is what we want to set as the top left
    H, W = dst.shape[:2]
    horLine = W/2
    verLine = H/2
    if closestToBottomRight[0] < horLine and closestToBottomRight[1] < verLine:  # top left
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
    elif closestToBottomRight[0] < horLine and closestToBottomRight[1] > verLine:  # bottom left
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
    elif closestToBottomRight[0] > horLine and closestToBottomRight[1] > verLine:  # bottom right
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
    return dst, smallCorners