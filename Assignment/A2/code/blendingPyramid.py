# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 14:59:50 2016

@author: zhouc
"""
import math
import numpy as np
import cv2
import time


def createPyramid(img, pyramidN):
    imagePyramid = list()
    gaussianPyramid = list()
    laplacePyramid = list()
    
    """
    TODO 1:
    Generate 3 pyramid: image, gaussian, laplacian

    add your code here
    """

    imagePyramid.append(img)
    for i in range(pyramidN):
        newimg = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        imagePyramid.append(newimg)
        
        newgaussian = cv2.resize(newimg, (img.shape[1], img.shape[0]))
        gaussianPyramid.append(newgaussian)
        
        laplacePyramid.append(img - newgaussian)
        img = newimg

    return imagePyramid, gaussianPyramid, laplacePyramid


start_time = time.time()
# config & input
Topic = 'apple'

backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_pyramid.png'

backImg = cv2.imread(backImageName) / 255.0
foreImg = cv2.imread(foreImageName) / 255.0
mask = cv2.imread(maskName) / 255.0

rows = backImg.shape[0]
cols = backImg.shape[1]
channels = backImg.shape[2]

if mask.ndim == 2:
    mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])

if mask.shape[2] == 1:
    mask = np.tile(mask, [1, 1, 3])
    
pyramidN = int(math.ceil(math.log(min(rows, cols) / 16, 2)))

# build pyramid
[imageFore, gaussianFore, laplaceFore] = createPyramid(foreImg, pyramidN)
[imageBack, gaussianBack, laplaceBack] = createPyramid(backImg, pyramidN)
[imageMask, gaussianMask, laplaceMask] = createPyramid(mask, pyramidN+1)

# combine laplacian pyramid
laplaceMerge = list()
"""
TODO 2 
Combine the laplacian pyramids of background and foreground 

add your code here
"""

for i in range(pyramidN):
    temp = np.ones(gaussianMask[i].shape)
    laplaceMerge.append(laplaceFore[i]*gaussianMask[i]+laplaceBack[i]*(temp-gaussianMask[i]))

# Combine the smallest scale image
"""
TODO 3  
Combine the smallest scale images of background and foreground 

add your code here
"""

temp = np.ones(gaussianMask[-1].shape)
imagemerged = imageFore[-1]*gaussianMask[-1]+imageBack[-1]*(temp-gaussianMask[-1])

# reconstruct & output
"""
TODO 4 
reconstruct the blending image by adding the gradient (in different scale) back to
the smallest scale image while upsampling

add your code here
"""

for i in range(pyramidN):
    imagemerged = cv2.resize(imagemerged, (laplaceMerge[-i-1].shape[1], laplaceMerge[-i-1].shape[0]))
    imagemerged = imagemerged + laplaceMerge[-i-1]

img = imagemerged

cv2.imshow('output', img)
cv2.waitKey(0)
cv2.imwrite(outputName, img * 255)
print("--- %s seconds ---" % (time.time() - start_time))
