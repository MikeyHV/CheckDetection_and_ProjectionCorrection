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

def returnCheck(image):
    imagec = proj3Helper.histeq(image)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    outputs = predictor(imagec)

    peopleMasks = []
    bookMasks = []
    isABookMask = False
    temp = outputs["instances"]
    finalImage = None

    if 73 in temp.pred_classes and 0 not in temp.pred_classes:
        # print("Skip me, go straight to the next step")
        orig = image.copy()
        image = imutils.resize(image, width=500)  ## if the image does not have any humans, resize, else dont
        ratio = orig.shape[1] / float(image.shape[1])

        edged = proj3Helper.getCanPipe(image)
        corners = proj3Helper.getRect(edged)
        dst = proj3Helper.fitScreenToRect(image, corners)
        finalImage, smallBox = proj3Helper.correctPerspectiveAndOrientationGivenCheck(dst)
    elif 73 in temp.pred_classes and 0 in temp.pred_classes:  # assuming only one book in the image
        bookmask = np.array(temp.pred_masks[list(temp.pred_classes).index(73)])
        indices = bookmask.astype(np.uint8)  # convert to an unsigned byte
        indices *= 255
        kernel = np.ones((7, 7), np.uint8)
        img_dilation = cv2.dilate(indices, kernel, iterations=20)
        approxCorners = getAppxCorner(img_dilation)
        finalImage = proj3Helper.fitScreenToRect(image, approxCorners)
    else:  # people in the image, no book segementation :(
        pass



def process_img(img_path):
    frame_orig = cv2.imread(img_path)
    ### Replace the code below to show only the check and apply transform.
    frame_result = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    ### Replace the code above.
    cv2.imshow("Original", frame_orig)
    cv2.imshow("Result", frame_result)
    cv2.waitKey(0)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Check prepartion project")
    parser.add_argument('--input_folder', type=str, default='samples', help='check images folder')
    
    args = parser.parse_args()
    input_folder = args.input_folder
   
    for check_img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, check_img)
        if img_path.lower().endswith(('.png','.jpg','.jpeg', '.bmp', '.gif', '.tiff')):
            process_img(img_path)
            