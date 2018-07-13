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
    
    row = img_set[0].shape[0]
    col = img_set[0].shape[1]

    # randomly choose N pixels
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


def tone_mapping_global(hdr):
    # compute intensity
    # return log(hdr/(hdr+1.0))
    return hdr/(hdr+1.0)


def tone_mapping_local(hdr):
    """
    Tone mapping
    :param hdr:input a 3 channel hdr image (radiance map)
    :param g_set:
    :return:
    """
    """
    TODO 4
    Follow spec to do tone mapping locally

    add your code here
    """

    # compute intensity by averaging color channels
    img_I = np.mean(hdr, axis=2)
    # img_I= 0.299 * hdr[:,:,0] + 0.587 * hdr[:,:,1] + 0.114 * hdr[:,:,2]

    # compute chrominance channels
    img_chrom = np.zeros(shape=(hdr.shape), dtype=np.float32)
    for i in range(3):
        img_chrom[:, :, i] = hdr[:, :, i] / img_I

    # compute log intensity
    img_L = np.log2(img_I)

    # bilateral filter
    # Note: change d and sigmaColor and sigmaSpace to produce best results
    d_ = 3
    sigColor_ = 20
    sigSpace_ = 3
    img_B = cv2.bilateralFilter(img_L, d=d_, sigmaColor=sigColor_, sigmaSpace=sigSpace_)

    # compute detail layer
    img_D = img_L - img_B

    # apply offset and scale to base
    o = np.amin(img_B)

    # adjust to produce good result
    dR = 10
    s = dR / (np.amax(img_B) - o)
    img_Bp = (img_B - o) * s

    # reconstruct log intensity
    img_O = np.exp2(img_Bp + img_D)

    # push back colors
    img_chromp = np.zeros(shape=(hdr.shape), dtype=np.float32)
    for i in range(3):
        img_chromp[:, :, i] = img_O * img_chrom[:, :, i]

    parm = 0.12
    result = np.power(img_chromp, parm)
    result = np.log2(result)

    return result

start_time = time.time()
# path to data
dataPath = '../data/'

topic = 'belg'
# topic = 'memorial'
# topic = '29'
# topic = 'table'
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
result = tone_mapping_local(hdr)
outputName = topic + '_output_local.jpg'
# result = tone_mapping_global(hdr)
# outputName = topic + '_output_global.jpg'

# show and save results
cv2.imshow('output', result)
cv2.waitKey(0)
result = np.clip(result*255, 0, 255).astype('uint8')
cv2.imwrite(outputName, result)

print("--- %s seconds ---" % (time.time() - start_time))


