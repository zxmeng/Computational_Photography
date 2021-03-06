import numpy as np
import scipy
from scipy.sparse import linalg
import cv2
import time


# config & input
start_time = time.time()
# Topic = 'snow'
Topic = 'notebook'

backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_poisson_tuto.png'

backImg = cv2.imread(backImageName) / 255.0
foreImg = cv2.imread(foreImageName) / 255.0
mask = cv2.imread(maskName) / 255.0

rows = backImg.shape[0]
cols = backImg.shape[1]
channels = backImg.shape[2]

alls = rows * cols * channels

# build matrix A and B

# counting the number of pixels inside the region to be pasted (white region in the mask)
mrows = mask.shape[0]
mcols = mask.shape[1]
mchan = mask.shape[2]
num = 0
for i in range(mrows):
    for j in range(mcols):
        if mchan == 1:
            if mask[i][j] == 1.0:
                num += 1
        else:
            if mask[i][j][0] == 1.0 and mask[i][j][1] == 1.0 and mask[i][j][2] == 1.0:
                num += 1

I = np.zeros(num*5*channels+alls)
J = np.zeros(num*5*channels+alls) 
S = np.zeros(num*5*channels+alls)
B = np.zeros(alls)
numRowsInA = 0

"""
TODO 5 
Construct matrix A & B

add your code here
"""
# construct matrix A
counter = -1
for i in range(rows):
    for j in range(cols):
        if channels == 1:
            if mask[i][j] == 1.0:
                # f(i,j)
                numRowsInA = i*cols + j
                counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA
                S[counter] = -4.0
                # f(i+1,j)
                if i < rows - 1:
                    counter += 1
                    I[counter] = numRowsInA
                    J[counter] = numRowsInA + cols
                    S[counter] = 1.0
                # f(i-1,j)
                if i > 0:
                    counter += 1
                    I[counter] = numRowsInA
                    J[counter] = numRowsInA - cols
                    S[counter] = 1.0
                # f(i,j+1)
                if j < cols -1:
                    counter += 1
                    I[counter] = numRowsInA
                    J[counter] = numRowsInA + 1
                    S[counter] = 1.0
                # f(i,j-1)
                if j > 0:
                    counter += 1
                    I[counter] = numRowsInA
                    J[counter] = numRowsInA - 1
                    S[counter] = 1.0   
            else:
            	numRowsInA = i*cols + j
                counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA
                S[counter] = 1.0
        else:
            if mask[i][j][0] == 1.0 and mask[i][j][1] == 1.0 and mask[i][j][2] == 1.0:
                for k in range(channels):
                    # f(i,j)
                    numRowsInA = i*cols*channels + j*channels + k
                    counter += 1
                    I[counter] = numRowsInA
                    J[counter] = numRowsInA
                    S[counter] = -4.0
                    # f(i+1,j)
                    if i < rows -1:
	                    counter += 1
	                    I[counter] = numRowsInA
	                    J[counter] = numRowsInA + cols*channels
	                    S[counter] = 1.0
                    # f(i-1,j)
                    if i > 0:
	                    counter += 1
	                    I[counter] = numRowsInA
	                    J[counter] = numRowsInA - cols*channels
	                    S[counter] = 1.0
                    # f(i,j+1)
                    if j < cols - 1:
	                    counter += 1
	                    I[counter] = numRowsInA
	                    J[counter] = numRowsInA + channels
	                    S[counter] = 1.0
                    # f(i,j-1)
                    if j > 0:
	                    counter += 1
	                    I[counter] = numRowsInA
	                    J[counter] = numRowsInA - channels
	                    S[counter] = 1.0
            else:
                for k in range(channels):
                    numRowsInA = i*cols*channels + j*channels + k
                    counter += 1
                    I[counter] = numRowsInA
                    J[counter] = numRowsInA
                    S[counter] = 1.0
           
# construct sparse matrix A           
numRowsInA = rows*cols*channels
A = scipy.sparse.coo_matrix((S, (I, J)), shape=(numRowsInA, alls))

# construct matrix B
for i in range(rows):
    for j in range(cols):
        if channels == 1:
            if mask[i][j] == 1.0:
                # dv/dx
                if i < rows - 1 and i > 0:
                    dvijx = (foreImg[i+1][j] - foreImg[i][j]) - (foreImg[i][j] - foreImg[i-1][j])
                    dvijx += (backImg[i+1][j] - backImg[i][j]) - (backImg[i][j] - backImg[i-1][j])
                elif i == rows - 1:
                    dvijx = (0.0 - foreImg[i][j]) - (foreImg[i][j] - foreImg[i-1][j])
                    dvijx += (0.0 - backImg[i][j]) - (backImg[i][j] - backImg[i-1][j])
                elif i == 0:
                    dvijx = (foreImg[i+1][j] - foreImg[i][j]) - (foreImg[i][j] - 0.0)
                    dvijx += (backImg[i+1][j] - backImg[i][j]) - (backImg[i][j] - 0.0)
                # dv/dy
                if j < cols - 1 and j > 0:
                    dvijy = (foreImg[i][j+1] - foreImg[i][j]) - (foreImg[i][j] - foreImg[i][j-1])
                    dvijy += (backImg[i][j+1] - backImg[i][j]) - (backImg[i][j] - backImg[i][j-1])
                elif j == cols - 1:
                    dvijy = (0.0 - foreImg[i][j]) - (foreImg[i][j] - foreImg[i][j-1])
                    dvijy += (0.0 - backImg[i][j]) - (backImg[i][j] - backImg[i][j-1])
                elif j == 0:
                    dvijy = (foreImg[i][j+1] - foreImg[i][j]) - (foreImg[i][j] - 0.0)
                    dvijy += (backImg[i][j+1] - backImg[i][j]) - (backImg[i][j] - 0.0)
                # div(v) = dv/dx + dv/dy
                B[i*cols+j] = dvijx + dvijy
            else:
            	B[i*cols+j] = backImg[i][j]
        else:
            if mask[i][j][0] == 1.0 and mask[i][j][1] == 1.0 and mask[i][j][2] == 1.0:
                for k in range(channels):
                    # dv/dx
                    if i < rows - 1 and i > 0:
                        dvijx = (foreImg[i+1][j][k] - foreImg[i][j][k]) - (foreImg[i][j][k] - foreImg[i-1][j][k])
                        dvijx += (backImg[i+1][j][k] - backImg[i][j][k]) - (backImg[i][j][k] - backImg[i-1][j][k])
                    elif i == rows - 1:
                        dvijx = (0.0 - foreImg[i][j][k]) - (foreImg[i][j][k] - foreImg[i-1][j][k])
                        dvijx += (0.0 - backImg[i][j][k]) - (backImg[i][j][k] - backImg[i-1][j][k])
                    elif i == 0:
                        dvijx = (foreImg[i+1][j][k] - foreImg[i][j][k]) - (foreImg[i][j][k] - 0.0)
                        dvijx += (backImg[i+1][j][k] - backImg[i][j][k]) - (backImg[i][j][k] - 0.0)
                    # dv/dy
                    if j < cols - 1 and j > 0:
                        dvijy = (foreImg[i][j+1][k] - foreImg[i][j][k]) - (foreImg[i][j][k] - foreImg[i][j-1][k])
                        dvijy += (backImg[i][j+1][k] - backImg[i][j][k]) - (backImg[i][j][k] - backImg[i][j-1][k])
                    elif j == cols - 1:
                        dvijy = (0.0 - foreImg[i][j][k]) - (foreImg[i][j][k] - foreImg[i][j-1][k])
                        dvijy += (0.0 - backImg[i][j][k]) - (backImg[i][j][k] - backImg[i][j-1][k])
                    elif j == 0:
                        dvijy = (foreImg[i][j+1][k] - foreImg[i][j][k]) - (foreImg[i][j][k] - 0.0)
                        dvijy += (backImg[i][j+1][k] - backImg[i][j][k]) - (backImg[i][j][k] - 0.0)
                    # div(v) = dv/dx + dv/dy
                    B[i*cols*channels+j*channels+k] = dvijx + dvijy
            else:
                for k in range(channels):
                    B[i*cols*channels+j*channels+k] = backImg[i][j][k]
                    
# solve Ax=B with least square
R = scipy.sparse.linalg.cg(A, B)

# make R an image & output
"""
TODO 6  
extract final result from R

add your code here
"""

img = np.zeros(backImg.shape)
for t in range(len(R[0])):
    k = t % channels
    r = t / channels
    j = r % cols
    i = r / cols
    if channels == 1:
        img[i][j] = R[0][t]
    else:
        img[i][j][k] = R[0][t]

cv2.imshow('output', img)
cv2.waitKey(0)
cv2.imwrite(outputName, img * 255)
print("--- %s seconds ---" % (time.time() - start_time))
