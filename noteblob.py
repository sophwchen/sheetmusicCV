import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import os

#blob detection and note/rest extraction
notedetect_minarea = 50
notedetect_maxarea = 1000

def morphFilterCircle(pimg, sz_reduce = 0, sz_expand = 0):
    kernel_reduce = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz_reduce, sz_reduce))
    result = cv2.dilate(np.array(pimg), kernel_reduce, iterations = 1)
    if sz_expand > 0:
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz_expand, sz_expand))
        result = cv2.erode(result, kernel_expand, iterations = 1)
    return result

def detectNoteheadBlobs(img, minarea, maxarea):
    
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = minarea
    params.maxArea = maxarea

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(np.array(img), keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return keypoints, im_with_keypoints

def extract_notes(imagefile):
    """
    accepts filepath to image of line of music
    uses opencv morphing and blob detection to locate clefs, notes, and rests
    returns list of boxes of clefs, notes, and rests
    """

    pim1 = Image.open(imagefile).convert('L')

    imtest = morphFilterCircle(pim1, 4, 2)
    imtest2 = morphFilterCircle(imtest,3, 5)

    keypoints, im_with_keypoints = detectNoteheadBlobs(imtest2, notedetect_minarea, notedetect_maxarea)
    print(len(keypoints))
    pim1 = np.array(pim1)
    coor = []
    for i in range(len(keypoints)):
        x, y = keypoints[i].pt
        x = int(x)
        y = int(y)
        coor.append((x,y))
    coor = sorted(coor, key=lambda x: x[0])

    delete = []
    middle = 90
    for i in range(len(coor)):
        if i < 1: continue
        if(i != 0 and abs(coor[i][0]-coor[i-1][0]) < 20):
            if(abs(middle - coor[i-1][0]) < abs(middle-coor[i][0])):
                delete.append(i)
            else:
                np.delete(coor, i-1)
                delete.append(i)

    newcoor = []
    for i in range(len(coor)):
        if i not in delete:
            newcoor.append(coor[i])


    notes = []
    for x,y in newcoor:
        notes.append(pim1[30:150, x-40:x+40])
    
    return notes