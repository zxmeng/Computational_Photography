# -*- coding: utf-8 -*-

import numpy as np
import caffe
import glob 
import cv2
import os


def extractY(img):
    # convert RGB to YCRCB first, then return the Y channel
    img = np.clip(img, 0, 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    img = img[:, :, 0] / 255.0
    return img


def computePSNR(imgA, imgB):
    rows = imgA.shape[0]
    cols = imgA.shape[1]
    # get Y channel first
    imgA = extractY(imgA)
    imgB = extractY(imgB)

    # compute MSE and PSNR
    mse = np.sum(np.square(imgA - imgB)) / float(rows*cols)
    psnr = 10.0 * np.log10(1.0 / mse)
    return psnr


# config paths
model_path = "examples/SRCNN/SRCNN_test.prototxt"

# testing images filepaths
folder = 'Test/Set5/'
filepaths = glob.glob(folder + '*.bmp')

# to record the osnr
dt = open("psnr_record.txt", 'a+')
count = 0
psnr_sum_bi = 0.0
psnr_sum_sr = 0.0
shrink = 12
padding = shrink/2
scale = 3

for num in range(11):
    # to use different models
    iteration = 1000*num + 20000
    param_path = "examples/SRCNN/SRCNN_iter_" + str(iteration) + ".caffemodel"
    count = 0
    psnr_sum_bi = 0.0
    psnr_sum_sr = 0.0

    # load model
    net = caffe.Net(model_path, param_path, caffe.TEST)

    dt.write("Iteration: " + str(iteration) + "\n")
    for f in filepaths:
        # read original image
        img_origin = cv2.imread(f)
        # padding
        img_input = cv2.copyMakeBorder(img_origin,padding,padding,padding,padding,cv2.BORDER_REFLECT)
        # get size for output
        sz = np.shape(img_input)
        rows = sz[0] - shrink
        cols = sz[1] - shrink
        out = np.zeros((rows, cols, 3))
        # bicubic interpolation
        img_input = cv2.resize(cv2.resize(img_input, (sz[1]/scale, sz[0]/scale), interpolation=cv2.INTER_AREA), (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)

        # super resolution for Y channel
        img_input_y = np.clip(img_input, 0, 255).astype('uint8')
    	img_input_y = cv2.cvtColor(img_input_y, cv2.COLOR_RGB2YCR_CB)
        ch = img_input_y[:,:,0] / 255.0

        net.blobs['data'].reshape(1, 1, sz[0], sz[1])
        net.blobs['data'].data[...] = np.reshape(ch, (1, 1, sz[0], sz[1]))
        net.forward()
        x = net.blobs['conv3'].data[...]
        out[:,:,0] = np.squeeze(x)

        # save outputs
        fn = os.path.splitext(os.path.basename(f))[0]
        # bicubic results
        img_in = img_input[shrink / 2 : sz[0] - shrink / 2, shrink / 2 : sz[1] - shrink / 2, :]
        cv2.imwrite(str(iteration) + '_' + fn + '_input.png', img_in)
        # super resolution results
        out[:,:,0] = out[:,:,0] * 255
        out[:,:,1] = img_input_y[shrink / 2 : sz[0] - shrink / 2, shrink / 2 : sz[1] - shrink / 2, 1]
        out[:,:,2] = img_input_y[shrink / 2 : sz[0] - shrink / 2, shrink / 2 : sz[1] - shrink / 2, 2]
        out = np.clip(out, 0, 255).astype('uint8')
    	img_out = cv2.cvtColor(out, cv2.COLOR_YCR_CB2RGB)
        cv2.imwrite(str(iteration) + '_' + fn + '_output.png', img_out)

        # compute and record PSNR
        psnr_bi = computePSNR(img_origin, img_in)
        psnr_sr = computePSNR(img_origin, img_out)
        psnr_sum_bi += psnr_bi
        psnr_sum_sr += psnr_sr
        count += 1
        dt.write(fn + " PSNR (BI): " + str(psnr_bi) + "\n")
        dt.write(fn + " PSNR (SR): " + str(psnr_sr) + "\n")

    # compute and record average PSNR
    dt.write("--- The average PSNR (BI) is: %s dB ---\n" % (psnr_sum_bi/count))
    dt.write("--- The average PSNR (SR) is: %s dB ---\n" % (psnr_sum_sr/count))
    dt.write("\n\n")

dt.close()


