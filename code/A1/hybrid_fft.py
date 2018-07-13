# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time

'''
visualize a hybrid image by progressively downsampling the image and
concatenating all of the images together.
'''
def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    
    output = hybrid_image
    cur_image = hybrid_image

    for i in range(2, scales + 1):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)
        cur_image = cv2.resize(cur_image, (0,0), fx=scale_factor, fy=scale_factor)
        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)
    
    return output


def pad_zero(image, kernel_row, kernel_col):
    img_row, img_col = image.shape[:2]
    a = (kernel_row-1)/2
    b = (kernel_col-1)/2
    new_row = img_row + 2*a
    new_col = img_col + 2*b
    
    if len(image.shape) == 3:
        new_img = np.zeros((new_row, new_col, 3), dtype=np.float32)
    else:
        new_img = np.zeros((new_row, new_col), dtype=np.float32)
        
    for i in range(img_row):
        for j in range(img_col):
            new_img[i+a][j+b] = image[i][j]
            
    return new_img
           
    
def pad_replicate(image, kernel_row, kernel_col):
    img_row, img_col = image.shape[:2]
    a = (kernel_row-1)/2
    b = (kernel_col-1)/2
    new_row = img_row + 2*a
    new_col = img_col + 2*b
    
    if len(image.shape) == 3:
        new_img = np.zeros((new_row, new_col, 3), dtype=np.float32)
    else:
        new_img = np.zeros((new_row, new_col), dtype=np.float32)
        
    for i in range(new_row):
        for j in range(new_col):
            if i < a:
                if j < b:
                    new_img[i][j] = image[0][0]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[0][img_col-1]
                else:
                    new_img[i][j] = image[0][j-b]
            elif i > (new_row-a-1):
                if j < b:
                    new_img[i][j] = image[img_row-1][0]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[img_row-1][img_col-1]
                else:
                    new_img[i][j] = image[img_row-1][j-b]
            else:
                if j < b:
                    new_img[i][j] = image[i-a][0]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[i-a][img_col-1]
                else:
                    new_img[i][j] = image[i-a][j-b]
                    
    return new_img
                    
                    
def pad_symmetric(image, kernel_row, kernel_col):
    img_row, img_col = image.shape[:2]
    a = (kernel_row-1)/2
    b = (kernel_col-1)/2
    new_row = img_row + 2*a
    new_col = img_col + 2*b
    
    if len(image.shape) == 3:
        new_img = np.zeros((new_row, new_col, 3), dtype=np.float32)
    else:
        new_img = np.zeros((new_row, new_col), dtype=np.float32)
        
    for i in range(new_row):
        for j in range(new_col):
            if i < a:
                if j < b:
                    new_img[i][j] = image[a-1-i][b-1-j]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[a-1-i][img_col+new_col-b-1-j]
                else:
                    new_img[i][j] = image[a-1-i][j-b]
            elif i > (new_row-a-1):
                if j < b:
                    new_img[i][j] = image[img_row+new_row-a-1-i][b-1-j]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[img_row+new_row-a-1-i][img_col+new_col-b-1-j]
                else:
                    new_img[i][j] = image[img_row+new_row-a-1-i][j-b]
            else:
                if j < b:
                    new_img[i][j] = image[i-a][b-1-j]
                elif j > (new_col-b-1):
                    new_img[i][j] = image[i-a][img_col+new_col-b-1-j]
                else:
                    new_img[i][j] = image[i-a][j-b]
                    
    return new_img


def my_imfilter_fft(image, kernel, padding_type):
    # detect whether it's grayscale or not
    if len(image.shape) == 3:
        gray_image = 0
    else:
        gray_image = 1
    
    # size of image and kernel
    img_row = image.shape[0]
    img_col = image.shape[1]
    kernel_row = len(kernel)
    kernel_col = len(kernel[0])
    fftsize = 1024
    
    # pad zeros
    if padding_type == 0:
        new_img = pad_zero(image, kernel_row, kernel_col)
    # pad replicated     
    elif padding_type == 1:
        new_img = pad_replicate(image, kernel_row, kernel_col)
    # pad symmetric             
    elif padding_type == 2:
        new_img = pad_symmetric(image, kernel_row, kernel_col)
    
    a = (img_row-1)/2
    b = (img_col-1)/2
    # padding kernel as the same size as padded image
    new_kernel = pad_zero(kernel, img_row, img_col)
    kernel_fft = np.fft.fft2(new_kernel, s=(fftsize, fftsize))
    
    filtered_img = np.zeros(image.shape, dtype=np.float32)
    # filtering, use FFT to do the convolution
    if gray_image == 0:
        # get the RGB value respectively
        img_r = new_img[:,:,0]
        img_g = new_img[:,:,1]
        img_b = new_img[:,:,2]
        # FFT transform
        img_r_fft = np.fft.fft2(img_r, s=(fftsize, fftsize))
        img_g_fft = np.fft.fft2(img_g, s=(fftsize, fftsize))
        img_b_fft = np.fft.fft2(img_b, s=(fftsize, fftsize))
        # convolution
        img_r_fil = img_r_fft * kernel_fft
        img_g_fil = img_g_fft * kernel_fft
        img_b_fil = img_b_fft * kernel_fft
        # inverse FFT transform
        img_r_ifft = np.fft.ifft2(img_r_fil)
        img_g_ifft = np.fft.ifft2(img_g_fil)
        img_b_ifft = np.fft.ifft2(img_b_fil)
        # remove the padding
        a += (kernel_row-1)
        b += (kernel_col-1)
        filtered_img[:,:,0] = np.abs(img_r_ifft[a:a+img_row, b:b+img_col])
        filtered_img[:,:,1] = np.abs(img_g_ifft[a:a+img_row, b:b+img_col])
        filtered_img[:,:,2] = np.abs(img_b_ifft[a:a+img_row, b:b+img_col])
    else:
        # if it's grayscale
        img_fft = np.fft.fft2(new_img, s=(fftsize, fftsize))
        img_fil = img_fft * kernel_fft
        img_ifft = np.fft.ifft2(img_fil)
        filtered_img = np.abs(img_ifft[a:a+img_row, b:b+img_col])
        
    return filtered_img


if __name__ == '__main__':
    start_time = time.time()
    
    image1 = cv2.imread('../data/dogbmp') / 255.0
    image2 = cv2.imread('../data/cat.bmp') / 255.0

    cutoff_frequency1 = 7
    cutoff_frequency2 = 7

    kernel1 = cv2.getGaussianKernel(cutoff_frequency1 * 4 + 1, cutoff_frequency1)
    kernel1 = cv2.mulTransposed(kernel1, False)

    kernel2 = cv2.getGaussianKernel(cutoff_frequency2 * 4 + 1, cutoff_frequency2)
    kernel2 = cv2.mulTransposed(kernel2, False)
    
    """
    YOUR CODE BELOW. Use my_imfilter to create 'low_frequencies' and
    'high_frequencies' and then combine them to create 'hybrid_image'
    """

    """
    TODO 2:
    Remove the high frequencies from image1 by blurring it. The amount of
    blur that works best will vary with different image pairs
    """
    # print "img1"
    low_frequencies = my_imfilter_fft(image1, kernel1, 0)

    """
    TODO 3:
    Remove the low frequencies from image2. The easiest way to do this is to
    subtract a blurred version of image2 from the original version of image2.
    This will give you an image centered at zero with negative values.
    """
    # print "img2"
    low_frequencies_temp = my_imfilter_fft(image2, kernel2, 0)
    high_frequencies = image2 - low_frequencies_temp

    """
    TODO 4: 
    Combine the high frequencies and low frequencies
    """
    hybrid_image = low_frequencies + high_frequencies

    """
    Visualize and save outputs
    """
    vis = vis_hybrid_image(hybrid_image)

    cv2.imshow("low", low_frequencies)
    cv2.imshow("high", high_frequencies + 0.5)
    cv2.imshow("vis", vis)
    cv2.waitKey(0)
    
    # to write the images, need to convert to 0~255
    low_frequencies = np.rint(low_frequencies * 255.0)
    high_frequencies = np.rint((high_frequencies + 0.5) * 255.0)
    hybrid_image = np.rint(hybrid_image * 255.0)
    vis = np.rint(vis * 255.0)

    cv2.imwrite('low_frequencies.jpg', low_frequencies)
    cv2.imwrite( 'high_frequencies.jpg', high_frequencies)
    cv2.imwrite('hybrid_image.jpg', hybrid_image)
    cv2.imwrite('hybrid_image_scales.jpg', vis)
    print("--- %s seconds ---" % (time.time() - start_time))
