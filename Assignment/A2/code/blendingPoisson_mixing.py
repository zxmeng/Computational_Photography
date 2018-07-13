import numpy as np
import scipy
from scipy.sparse import linalg
import cv2
import time


# config & input
start_time = time.time()
# Topic = 'snow'
Topic = 'sky'

backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_poisson_mixed.png'

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

I = np.zeros(num*5*channels)
J = np.zeros(num*5*channels) 
S = np.zeros(num*5*channels)
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
                    if i < rows -1 and mask[i+1][j][0] == 1.0 and mask[i+1][j][1] == 1.0 and mask[i+1][j][2] == 1.0:
                        counter += 1
                        I[counter] = numRowsInA
                        J[counter] = numRowsInA + cols*channels
                        S[counter] = 1.0
                    # f(i-1,j)
                    if i > 0 and mask[i-1][j][0] == 1.0 and mask[i-1][j][1] == 1.0 and mask[i-1][j][2] == 1.0:
                        counter += 1
                        I[counter] = numRowsInA
                        J[counter] = numRowsInA - cols*channels
                        S[counter] = 1.0
                    # f(i,j+1)
                    if j < cols - 1 and mask[i][j+1][0] == 1.0 and mask[i][j+1][1] == 1.0 and mask[i][j+1][2] == 1.0:
                        counter += 1
                        I[counter] = numRowsInA
                        J[counter] = numRowsInA + channels
                        S[counter] = 1.0
                    # f(i,j-1)
                    if j > 0 and mask[i][j-1][0] == 1.0 and mask[i][j-1][1] == 1.0 and mask[i][j-1][2] == 1.0:
                        counter += 1
                        I[counter] = numRowsInA
                        J[counter] = numRowsInA - channels
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
                    dfx1 = backImg[i+1][j] - backImg[i][j]
                    dgx1 = foreImg[i+1][j] - foreImg[i][j]
                    dfx2 = backImg[i][j] - backImg[i-1][j]
                    dgx2 = foreImg[i][j] - foreImg[i-1][j]
                elif i == rows - 1:
                    dfx1 = 0.0 - backImg[i][j]
                    dgx1 = 0.0 - foreImg[i][j]
                    dfx2 = backImg[i][j] - backImg[i-1][j]
                    dgx2 = foreImg[i][j] - foreImg[i-1][j]
                elif i == 0:
                    dfx1 = backImg[i+1][j] - backImg[i][j]
                    dgx1 = foreImg[i+1][j] - foreImg[i][j]
                    dfx2 = backImg[i][j] - 0.0
                    dgx2 = foreImg[i][j] - 0.0
                # dv/dy

                if j < cols - 1 and j > 0:
                    dfy1 = backImg[i][j+1] - backImg[i][j]
                    dgy1 = foreImg[i][j+1] - foreImg[i][j]
                    dfy2 = backImg[i][j] - backImg[i][j-1]
                    dgy2 = foreImg[i][j] - foreImg[i][j-1]
                elif j == cols - 1:
                    dfy1 = 0.0 - backImg[i][j]
                    dgy1 = 0.0 - foreImg[i][j]
                    dfy2 = backImg[i][j] - backImg[i][j-1]
                    dgy2 = foreImg[i][j] - foreImg[i][j-1]
                elif j == 0:
                    dfy1 = backImg[i][j+1] - backImg[i][j]
                    dgy1 = foreImg[i][j+1] - foreImg[i][j]
                    dfy2 = backImg[i][j] - 0.0
                    dgy2 = foreImg[i][j] - 0.0
                # choose the larger gradient
                if abs(dfx1) > abs(dgx1):
                    vx1 = dfx1
                else:
                    vx1 = dgx1
                if abs(dfx2) > abs(dgx2):
                    vx2 = dfx2
                else:
                    vx2 = dgx2
                if abs(dfy1) > abs(dgy1):
                    vy1 = dfy1
                else:
                    vy1 = dgy1
                if abs(dfy2) > abs(dgy2):
                    vy2 = dfy2
                else:
                    vy2 = dgy2
                dvijx = vx1 - vx2
                dvijy = vy1 - vy2
                # div(v) = dv/dx + dv/dy
                B[i*cols+j] = dvijx + dvijy
        else:
            if mask[i][j][0] == 1.0 and mask[i][j][1] == 1.0 and mask[i][j][2] == 1.0:
                for k in range(channels):
                    # dv/dx
                    if i < rows - 1 and i > 0:
                        dfx1 = backImg[i+1][j][k] - backImg[i][j][k]
                        dgx1 = foreImg[i+1][j][k] - foreImg[i][j][k]
                        dfx2 = backImg[i][j][k] - backImg[i-1][j][k]
                        dgx2 = foreImg[i][j][k] - foreImg[i-1][j][k]
                    elif i == rows - 1:
                        dfx1 = 0.0 - backImg[i][j][k]
                        dgx1 = 0.0 - foreImg[i][j][k]
                        dfx2 = backImg[i][j][k] - backImg[i-1][j][k]
                        dgx2 = foreImg[i][j][k] - foreImg[i-1][j][k]
                    elif i == 0:
                        dfx1 = backImg[i+1][j][k] - backImg[i][j][k]
                        dgx1 = foreImg[i+1][j][k] - foreImg[i][j][k]
                        dfx2 = backImg[i][j][k] - 0.0
                        dgx2 = foreImg[i][j][k] - 0.0
                    # dv/dy
                    if j < cols - 1 and j > 0:
                        dfy1 = backImg[i][j+1][k] - backImg[i][j][k]
                        dgy1 = foreImg[i][j+1][k] - foreImg[i][j][k]
                        dfy2 = backImg[i][j][k] - backImg[i][j-1][k]
                        dgy2 = foreImg[i][j][k] - foreImg[i][j-1][k]
                    elif j == cols - 1:
                        dfy1 = 0.0 - backImg[i][j][k]
                        dgy1 = 0.0 - foreImg[i][j][k]
                        dfy2 = backImg[i][j][k] - backImg[i][j-1][k]
                        dgy2 = foreImg[i][j][k] - foreImg[i][j-1][k]
                    elif j == 0:
                        dfy1 = backImg[i][j+1][k] - backImg[i][j][k]
                        dgy1 = foreImg[i][j+1][k] - foreImg[i][j][k]
                        dfy2 = backImg[i][j][k] - 0.0
                        dgy2 = foreImg[i][j][k] - 0.0
                    # choose the larger gradient
                    if abs(dfx1) > abs(dgx1):
                        vx1 = dfx1
                    else:
                        vx1 = dgx1
                    if abs(dfx2) > abs(dgx2):
                        vx2 = dfx2
                    else:
                        vx2 = dgx2
                    if abs(dfy1) > abs(dgy1):
                        vy1 = dfy1
                    else:
                        vy1 = dgy1
                    if abs(dfy2) > abs(dgy2):
                        vy2 = dfy2
                    else:
                        vy2 = dgy2
                    dvijx = vx1 - vx2
                    dvijy = vy1 - vy2
                    # div(v) = dv/dx + dv/dy
                    B[i*cols*channels+j*channels+k] = dvijx + dvijy
                    
# solve Ax=B with least square
R = scipy.sparse.linalg.cg(A, B)

# make R an image & output
"""
TODO 6  
extract final result from R

add your code here
"""

# get gradient matrix from R
gradient = np.zeros(backImg.shape)
for t in range(len(R[0])):
    k = t % channels
    r = t / channels
    j = r % cols
    i = r / cols
    if channels == 1:
        gradient[i][j] = R[0][t]
    else:
        gradient[i][j][k] = R[0][t]
        
# construct new image by gradient matrix and back image
img = np.zeros(backImg.shape)
for i in range(rows):
    for j in range(cols):
        if channels == 1:
            if mask[i][j] == 0.0:
                img[i][j] = backImg[i][j]
            else:
                img[i][j] = backImg[i][j] + gradient[i][j]
        else:
            if mask[i][j][0] == 0.0 and mask[i][j][1] == 0.0 and mask[i][j][2] == 0.0:
                for k in range(channels):
                    img[i][j][k] = backImg[i][j][k]
            else:
                for k in range(channels):
                    img[i][j][k] = backImg[i][j][k] + gradient[i][j][k]

cv2.imshow('output', img)
cv2.waitKey(0)
cv2.imwrite(outputName, img * 255)
print("--- %s seconds ---" % (time.time() - start_time))
