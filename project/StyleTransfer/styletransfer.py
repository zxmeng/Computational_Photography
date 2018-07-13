# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:52:41 2016

@author: zhouc
"""

# library imports
import caffe
import cv2
import time

# local imports
from style import StyleTransfer

start_time = time.time()

caffe.set_mode_cpu()

style_img_path = 'images/style/starry_night.jpg'
content_img_path = 'images/content/nanjing.jpg'

img_style = caffe.io.load_image(style_img_path)
img_content = caffe.io.load_image(content_img_path)

n_iter = 512
ratio = 1e4
args = {"length": 640, "ratio": ratio, "n_iter": n_iter, "init": "content"}

st = StyleTransfer()
st.transfer_style(img_style, img_content, **args)
img_out = st.get_generated()
img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

cv2.imwrite("output_nj_sn_" + str(n_iter) + "_" + str(ratio) + ".png", img_out*255)
print("--- %s seconds ---" % (time.time() - start_time))

