import numpy as np
import scipy
from scipy.sparse import linalg
import cv2
import time


# config & input
start_time = time.time()
Topic = 'bt42'

inputName = Topic + '.png'
outputName = Topic + '_out_poisson_gray.png'
outputName_darker = Topic + '_out_poisson_gray_darker.png'
grayImgName = Topic + '_out_gray.png'

# read source image and get the hsv/grayscal version
inputImg = cv2.imread(inputName)
img_hsv = cv2.cvtColor(inputImg, cv2.COLOR_RGB2HSV)
img_gray = cv2.cvtColor(inputImg, cv2.COLOR_RGB2GRAY)
cv2.imwrite(grayImgName, img_gray)

# treat the value channel as background image in poisson blending
# treat the saturation channel as foreground image in poisson blending
backImg = np.zeros((inputImg.shape[0], inputImg.shape[1]))
foreImg = np.zeros((inputImg.shape[0], inputImg.shape[1]))
mask = np.ones((inputImg.shape[0], inputImg.shape[1]))
# v channel
backImg = img_hsv[:,:,2] / 255.0
# s channel
foreImg = img_hsv[:,:,1] / 255.0

# the rest is poisson blending process
rows = backImg.shape[0]
cols = backImg.shape[1]
alls = rows * cols

# build matrix A and B

# counting the number of pixels inside the region to be pasted (white region in the mask)
mrows = mask.shape[0]
mcols = mask.shape[1]
num = 0
for i in range(mrows):
    for j in range(mcols):
        if mask[i][j] == 1.0:
                num += 1

I = np.zeros(num*5)
J = np.zeros(num*5) 
S = np.zeros(num*5)
B = np.zeros(alls)
numRowsInA = 0

# construct matrix A
counter = -1
for i in range(rows):
    for j in range(cols):
        if mask[i][j] == 1.0:
            # f(i,j)
            numRowsInA = i*cols + j
            counter += 1
            I[counter] = numRowsInA
            J[counter] = numRowsInA
            S[counter] = -4.0
            # f(i+1,j)
            if i < rows - 1 and mask[i+1][j] == 1.0:
                counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA + cols
                S[counter] = 1.0
            # f(i-1,j)
            if i > 0 and mask[i-1][j] == 1.0:
                counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA - cols
                S[counter] = 1.0
            # f(i,j+1)
            if j < cols -1 and mask[i][j+1] == 1.0:
                counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA + 1
                S[counter] = 1.0
            # f(i,j-1)
            if j > 0 and mask[i][j-1] == 1.0:
                counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA - 1
                S[counter] = 1.0    
    
# construct sparse matrix A             
numRowsInA = rows*cols
A = scipy.sparse.coo_matrix((S, (I, J)), shape=(numRowsInA, alls))

# construct matrix B
for i in range(rows):
    for j in range(cols):
        if mask[i][j] == 1.0:
            # dv/dx
            if i < rows - 1 and i > 0:
                dvijx = (foreImg[i+1][j] - foreImg[i][j]) - (foreImg[i][j] - foreImg[i-1][j])
            elif i == rows - 1:
                dvijx = (0.0 - foreImg[i][j]) - (foreImg[i][j] - foreImg[i-1][j])
            elif i == 0:
                dvijx = (foreImg[i+1][j] - foreImg[i][j]) - (foreImg[i][j] - 0.0)
            # dv/dy
            if j < cols - 1 and j > 0:
                dvijy = (foreImg[i][j+1] - foreImg[i][j]) - (foreImg[i][j] - foreImg[i][j-1])
            elif j == cols - 1:
                dvijy = (0.0 - foreImg[i][j]) - (foreImg[i][j] - foreImg[i][j-1])
            elif j == 0:
                dvijy = (foreImg[i][j+1] - foreImg[i][j]) - (foreImg[i][j] - 0.0)
            # div(v) = dv/dx + dv/dy
            B[i*cols+j] = dvijx + dvijy
         
# solve Ax=B with least square
R = scipy.sparse.linalg.cg(A, B)

# make R an image & output
# get gradient matrix from R
gradient = np.zeros(backImg.shape)
for t in range(len(R[0])):
    j = t % cols
    i = t / cols
    gradient[i][j] = R[0][t]

# construct new image by gradient matrix and back image
img = np.zeros(backImg.shape)
for i in range(rows):
    for j in range(cols):
        if mask[i][j] == 0.0:
            img[i][j] = backImg[i][j]
        else:
            img[i][j] = backImg[i][j] + gradient[i][j]

# output
cv2.imshow('output', img)
cv2.imshow('output_darker', img - 0.5)
cv2.waitKey(0)
cv2.imwrite(outputName, img * 255)
cv2.imwrite(outputName_darker, (img - 0.5) * 255)
print("--- %s seconds ---" % (time.time() - start_time))
