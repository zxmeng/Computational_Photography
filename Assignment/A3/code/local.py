import os
import numpy as np
import scipy
from scipy.sparse import linalg
import cv2
import random
import math
import matplotlib.pyplot as plt
import time


def weight_func(z):
    # weighting function
    return 127.5 - np.abs(1.0 * z - 127.5)


def select_Z_rand(img_set, N):
    """
    TODO 1
    Select a subset of pixels to estimate g()
    :param img_set:a list of total sequence
    :param N:number of pixels to select
    :param channel:the channel of images to select. Should be [0,1,2]
    :return:Z
    """
    # randomly choose N pixels
    row = img_set[0].shape[0]
    col = img_set[0].shape[1]

    index = np.zeros((N, 2), dtype=np.int)
    index[:, 0] = random.sample(xrange(row), N)
    index[:, 1] = random.sample(xrange(col), N)

    P = len(img_set)
    Z = np.zeros((3, N, P), dtype=np.int)
    for i in range(N):
        for j in range(P):
            for k in range(3):
                Z[k][i][j] = img_set[j][index[i][0]][index[i][1]][k]

    return Z


def radiance_map_construction(Z, exposure, lam):
    # min / max / mid value for Z
    Zmin = 0.0
    Zmax = 255.0
    Zmid = 128.0

    # size of each variable
    N, F = Z.shape
    n = 256

    # construct A and b
    total = N*F*2 + n*3 - 2 + 1
    I = np.zeros(total, dtype=np.int)
    J = np.zeros(total, dtype=np.int)
    S = np.zeros(total, dtype=np.float32)
    """
    TODO 2
    Construct matrix A & b

    add your code here
    """

    # build the weight matrix
    w = np.array(range(n))
    w = weight_func(w)

    # build A
    # add data terms
    counter = -1
    for i in range(N):
        for j in range(F):
            counter += 1
            I[counter] = i*F + j
            J[counter] = Z[i][j]
            S[counter] = w[Z[i][j]]

            counter += 1
            I[counter] = i*F + j
            J[counter] = n + i
            S[counter] = -1.0 * w[Z[i][j]]

    # add smooth terms
    lam_sqrt = math.sqrt(lam)
    for k in range(n-1):
            counter += 1
            I[counter] = N*F + k + 1
            J[counter] = k + 1
            S[counter] = -2.0 * lam_sqrt * w[k+1]
        
            counter += 1
            I[counter] = N*F + k + 1
            J[counter] = k 
            S[counter] = lam_sqrt * w[k+1]
        
            counter += 1
            I[counter] = N*F + k + 1
            J[counter] = k + 2
            S[counter] = lam_sqrt * w[k+1]

    # add g(128) = 0
    counter += 1
    I[counter] = N*F + n
    J[counter] = Zmid
    S[counter] = 1

    A = scipy.sparse.coo_matrix((S, (I, J)), shape=(N * F + n + 1, n + N))

    # build b
    B = np.zeros(shape=(N * F + n + 1), dtype=np.float32)
    for i in range(N):
        for j in range(F):
            B[i*F+j] = w[Z[i][j]] * math.log(exposure[j])

    # solve Ax=B with least square
    x = scipy.sparse.linalg.lsqr(A, B)

    # the first 256 elements belong to g
    g = x[0][0:n]

    return g


def createPyramid(img, pyramidN):
    # create gaussian pyramid for img
    gaussianPyramid = list()
    for i in range(pyramidN):
        newimg = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        newgaussian = cv2.resize(newimg, (img.shape[1], img.shape[0]))
        gaussianPyramid.append(newgaussian)
        img = newimg

    return gaussianPyramid


def attenue_parm(img):
    rows = img.shape[0]
    cols = img.shape[1]

    pyramidN = int(math.ceil(math.log(min(rows, cols) / 16, 2)))
    img_list = createPyramid(img, pyramidN)

    img_gm_list = list()

    kernel_x = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    kernel_y = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    # compute gradient magnitude at each level
    for i in range(pyramidN):
        # compute gradient
        img_g_x = cv2.filter2D(img_list[i], -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
        img_g_y = cv2.filter2D(img_list[i], -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
        img_g_x = img_g_x/np.exp2(i+1)
        img_g_y = img_g_y/np.exp2(i+1)

        # compute gradient magnitude
        img_gm = np.sqrt(np.power(img_g_x, 2) + np.power(img_g_y, 2)).astype(np.float32)
        img_gm_list.append(img_gm)
        
    # compute scaling factor at each level
    parm_list = list()
    for i in range(pyramidN):
        img_grad = img_gm_list[i]
        img_grad[img_grad == 0] = 0.0001

        alpha = 0.01 * np.mean(img_grad)
        beta = 0.9
        parm = np.power(img_grad/alpha, beta-1).astype(np.float32)
        parm_list.append(parm)

    # compute attenuing factor
    for i in range(pyramidN):
        d = pyramidN - 1 - i
        if i == 0:
            theta = parm_list[d]
        else:
            theta = cv2.resize(theta, (img_list[d].shape[1], img_list[d].shape[0]))
            theta = theta * parm_list[d]

    return theta


def tone_mapping_local2(hdr):
    # compute initial intensity
    img_I = np.mean(hdr, axis=2)

    # compute chrominance channels
    img_chrom = np.zeros(shape=(hdr.shape), dtype=np.float32)
    for i in range(3):
        img_chrom[:, :, i] = hdr[:, :, i] / img_I

    # compute logI -> H
    img_L = np.log2(img_I)

    # compute the attenuing factor
    theta = attenue_parm(img_L)

    # compute gradient
    kernel_x = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    kernel_y = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    img_Lg_x = cv2.filter2D(img_L, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    img_Lg_y = cv2.filter2D(img_L, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)

    img_Lgp_xp = img_Lg_x * theta
    img_Lgp_yp = img_Lg_y * theta

    # compute divergence of gradient
    kernel_x = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    kernel_y = np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    img_Lgp_div_x = cv2.filter2D(img_Lgp_xp, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    img_Lgp_div_y = cv2.filter2D(img_Lgp_yp, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    img_Lgp_div = img_Lgp_div_x + img_Lgp_div_y

    row = hdr.shape[0]
    col = hdr.shape[1]
    alls = row*col

    I = np.zeros(alls*5, dtype=np.int)
    J = np.zeros(alls*5, dtype=np.int)
    S = np.zeros(alls*5, dtype=np.float32)

    img_div_mean = np.mean(img_Lgp_div).astype(np.float32)
    img_Lgp_div = img_Lgp_div - img_div_mean
    B = np.reshape(img_Lgp_div, (alls))

    counter = -1
    numRowsInA = 0
    for i in range(row):
        for j in range(col):
            numRowsInA = i*col + j
            counter += 1
            I[counter] = numRowsInA
            J[counter] = numRowsInA
            S[counter] = -4.0

            if i > 0:
            	counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA - col
                S[counter] = 1.0
            if i < row - 1:
            	counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA + col
                S[counter] = 1.0
            if j > 0:
            	counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA - 1
                S[counter] = 1.0
            if j < col - 1:
            	counter += 1
                I[counter] = numRowsInA
                J[counter] = numRowsInA + 1
                S[counter] = 1.0

    A = scipy.sparse.coo_matrix((S, (I, J)), shape=(alls, alls))

    # solve Ax=B with least square
    R = scipy.sparse.linalg.cg(A, B)
    R = np.reshape(R[0], (row, col))

    img_O = np.exp2(R)

    ratio = 0.15
    img_chrom = np.power(img_chrom, ratio).astype(np.float32)

    # push back colors
    img_chromp = np.zeros(shape=(hdr.shape), dtype=np.float32)
    for i in range(3):
        img_chromp[:, :, i] = img_O * img_chrom[:, :, i]

    pparm = 3
    tparm = 0.25
    result = np.power(img_chromp, pparm) * tparm

    return result


start_time = time.time()
# path to data
dataPath = '../data/'
topic = 'belg'
# topic = 'memorial'
# topic = '29'
# topic = 'table' 
# topic = 'cloud'
# topic = 'garage'

# file names and exposure time for case 1
imgList = []
for x in range(1,10):
    name = 'case1/belg00' + str(x) + '.jpg'
    imgList.append(name)
exposure = [1.0/1000, 1.0/500, 1.0/250, 1.0/125, 1.0/60, 1.0/30, 1.0/15, 1.0/8, 1.0/4]

# file names and exposure time for case 2
# imgList = []
# for x in range(1,10):
#     name = 'case2/memorial006' + str(x) + '.png'
#     imgList.append(name)
# for x in range(7):
#     name = 'case2/memorial007' + str(x) + '.png'
#     imgList.append(name)
# exposure = [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64, 1.0/128, 1.0/256, 1.0/512, 1.0/1024]

# file names and exposure time for case 3
# imgList = []
# for x in range(1, 9):
#     name = 'case3/' + str(x) + '.29.png'
#     imgList.append(name)
# exposure = [0.003, 0.006, 0.012, 0.024, 0.048, 0.096, 0.192, 0.384]

# file names and exposure time for case 4
# imgList = []
# for x in range(1,14):
#     name = 'case4/img' + str(x) + '.jpg'
#     imgList.append(name)
# exposure = [13.0, 10.0, 4.0, 3.2, 1.0, 0.8, 0.3, 0.25, 0.01666666, 0.0125, 0.003125, 0.0025, 0.001]

# file names and exposure time for case 6
# imgList = []
# for x in range(1,10):
#     name = 'case6/garage0' + str(x) + '.jpg'
#     imgList.append(name)
# exposure = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]


# read all images into a list
img_set = []
for imgFile in imgList:
    # read one image
    img = cv2.imread(os.path.join(dataPath, imgFile)).astype(np.float32)
    img_set.append(img)

# estimate g for each channel
lam = 0.2
N = 300

# to save g for each image channel
g_set = np.zeros(shape=([3, 256]), dtype=np.float32)

# to save hdr result
hdr = np.zeros(shape=(img_set[0].shape), dtype=np.float32)
row = img_set[0].shape[0]
col = img_set[0].shape[1]
P = len(img_set)
y = np.arange(256)

"""
TODO 3
Compute the whole radiance map

add your code here
"""

# recover radiance map
Z = select_Z_rand(img_set, N)
for ich in range(3):
    g = radiance_map_construction(Z[ich], exposure, lam)
    g_set[ich] = g
    plt.plot(g, y)

# recover HDR radiance values
# convert image list to numpy array
img_set = np.array(img_set, dtype=np.int)

# compute weight for images
weight = weight_func(img_set)

# compute radiance values
g_value = np.zeros(shape=([P, row, col, 3]), dtype=np.float32)
for i in range(3):
    g_value[:, :, :, i] = g_set[i][img_set[:, :, :, i]]
for p in range(P):
    g_value[p] -= math.log(exposure[p])
g_value = g_value * weight

temp_n = np.sum(g_value, axis=0)
temp_d = np.sum(weight, axis=0)
# eliminate zeros
temp_d[temp_d == 0] = 0.001
hdr = np.exp(temp_n/temp_d).astype(np.float32)

# show and save radiance map
name = 'rad_map_' + str(N) + '_' + str(lam) + '.png'
plt.savefig(name)
plt.show()

# tone mapping
result = tone_mapping_local2(hdr)
outputName = topic + '_output_local2.jpg'

# show and save results
cv2.imshow('output', result)
cv2.waitKey(0)
result = np.clip(result*255, 0, 255).astype('uint8')
cv2.imwrite(outputName, result)

print("--- %s seconds ---" % (time.time() - start_time))


