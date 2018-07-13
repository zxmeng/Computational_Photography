# -*- coding: utf-8 -*-
import h5py
import numpy as np
import glob 
import cv2
import random


def modcrop(img, modulo):
    sz = np.shape(img)
    sz = sz - np.mod(sz, modulo)
    return img[0 : sz[0] - 1, 0 : sz[1] - 1]


def store2hdf5(filename, data, labels):

    f = h5py.File(filename, "w")

    f.create_dataset('data', data=data)
    f.create_dataset('label', data=labels)
    f.close()


folder = 'Train/'
savepath = 'train.h5'

# the size of input for cnn, 11*11
size_input = 11
# the size of output for cnn, 19*19
size_label = 19

padwidth = (size_label-size_input)/2

scale = 3
# the stride for cropping training data
stride = 4

max_sample = 100000

data = np.reshape(np.array([]), (size_input, size_input, 1, 0))
label = np.reshape(np.array([]), (size_label, size_label, 1, 0))

data = np.zeros((size_input, size_input, 1, max_sample))
label = np.zeros((size_label, size_label, 1, max_sample))

padding = abs(size_input - size_label) / 2

filepaths = glob.glob(folder + '*.bmp')
count = -1
for f in filepaths:
    # read the image
    image = cv2.imread(f)
    # convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    # get the Y channel
    image = image[:, :, 0] / 255.0

    im_label = modcrop(image, scale)
    sz = np.shape(im_label)
    # downsampling and then upsampling to get the low resolution input
    im_input = cv2.resize(im_label, (sz[1]/scale, sz[0]/scale), interpolation=cv2.INTER_AREA)
    sz_new = im_input.shape

    """
    TODO #1:
    Generate training data pairs. 
    """      
    temp_r = 0
    temp_c = 0
    while temp_c+size_input<sz_new[1]:
        while temp_r+size_input<sz_new[0]:
            count += 1

            loc_r = scale * (temp_r + np.floor((size_input - 1)/2)) - np.floor((size_label + scale)/2 - 1)
            loc_c = scale * (temp_c + np.floor((size_input - 1)/2)) - np.floor((size_label + scale)/2 - 1)
            
            data[:,:,0,count] = im_input[temp_r : size_input + temp_r, temp_c : size_input + temp_c]
            label[:,:,0,count] = im_label[loc_r : size_label + loc_r, loc_c : size_label + loc_c]

            temp_r += stride
            if (count+1) == max_sample:
                break
        if (count+1) == max_sample:
            break
        temp_r = 0
        temp_c += stride
    if (count+1) == max_sample:
            break 

data = data[:,:,:,0:count]
label = label[:,:,:,0:count]

"""
TODO #1:
Randomly permute the data pairs.
"""
data = np.transpose(data, (3, 2, 1, 0))
label = np.transpose(label, (3, 2, 1, 0))

neworder = random.sample(range(count), count)
newdata = data[neworder]
newlabel = label[neworder]

store2hdf5(savepath, newdata, newlabel)

