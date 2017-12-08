#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:06:20 2017

@author: hjxu
"""#! coding=utf-8
caffe_root = '/home/hjxu/caffe-master/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import scipy
import cv2
import skimage
from skimage import measure,morphology
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
caffe.set_mode_gpu()
caffe.set_device(0)

start=time.time()

def get_bbox(cont_img, rgb_image=None):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgb_contour


def read_wsi_tumor(wsi_path):
    try:
        wsi_image = OpenSlide(wsi_path)
        m, n = wsi_image.dimensions
        level_used = 8
        m, n = int(m / 256), int(n / 256)
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                   wsi_image.level_dimensions[level_used]))
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

    return wsi_image, rgb_image, level_used,m,n

def find_roi_bbox(rgb_image):
    # hsv -> 3 channel
    rgb_image = rgb_image[:,:,:3]
#    imgGray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) 
#    threshold,imgOtsu = cv2.threshold(imgGray,200,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    lower_red = np.array([10, 10, 10])
    upper_red = np.array([220, 220, 220])
    mask = cv2.inRange(rgb_image, lower_red, upper_red)
    plt.imshow(mask)
    plt.show()
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    plt.imshow(image_open)
    plt.show()
    cv2.imwrite("/home/hjxu/breast_project/cc_open.png",image_open)
    bounding_boxes, rgb_contour = get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open


start=time.time()
root = '/home/hjxu/caffe_examples/metastatic/Alexnet/'
deploy =root + 'profile/deploy.prototxt'
model = '/home/hjxu/caffe_examples/metastatic/Alexnet/Alexnet__iter_150000.caffemodel'
mean='/home/hjxu_disk/lmdb/train_mean.npy'


WSI = '/media/hjxu/Elements/2017-01-13/2017-01-13 19.04.36.ndpi'
wsi_image, rgb_image,level,m,n=read_wsi_tumor(WSI)
bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
image_heat_save = np.zeros((n,m))

net = caffe.Net(deploy, model, caffe.TEST)


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
print('%s Classification is in progress' % WSI)
for bounding_box in bounding_boxes:
    b_x_start = int(bounding_box[0])
    b_y_start = int(bounding_box[1])
    b_x_end = int(bounding_box[0]) + int(bounding_box[2])
    b_y_end = int(bounding_box[1]) + int(bounding_box[3])
    #        X = np.random.random_integers(b_x_start, high=b_x_end, size=500 )
    #        Y = np.random.random_integers(b_y_start, high=b_y_end, size=int((b_y_end-b_y_start)//2+1 ))
    col_cords = np.arange(b_x_start, b_x_end)
    row_cords = np.arange(b_y_start, b_y_end)
    mag_factor = 256
    #        for x, y in zip(X, Y):
    #            if int(tumor_gt_mask[y, x]) != 0:
    for x in col_cords:
        for y in row_cords:
            if int(image_open[y, x]) != 0:
                x_large = x * mag_factor
                y_large = y * mag_factor
                patch = wsi_image.read_region((x_large, y_large), 0, (256, 256))
                img_tmp = skimage.img_as_float(np.array(patch))
                img1 = np.tile(img_tmp, (1, 1, 3))
                img2 = img1[:, :, :3]
                # patch.save('/home/hjxu/caffe_examples/metastatic/Alexnet/temp/temp.png', 'PNG')
                # tile = tile = np.asarray(Image.open('/home/hjxu/caffe_examples/metastatic/Alexnet/temp/temp.png'))
                # img = tile[:, :, 0:3]
                # scipy.misc.imsave('/home/hjxu/caffe_examples/metastatic/Alexnet/temp1.png', img)
                # img2 = caffe.io.load_image('/home/hjxu/caffe_examples/metastatic/Alexnet/temp1.png')
                net.blobs['data'].data[...] = transformer.preprocess('data', img2)
                out = net.forward()
                prob = out['prob'][0][0]
                image_heat_save[y, x] = prob
np.save("/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict2/Tumor_110.npy", image_heat_save)
plt.imshow(image_heat_save)
plt.show()
scipy.misc.imsave('/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict2/Tumor_110.tif', image_heat_save)
end = time.time()
print ('run time%s'%(end-start))
print('has done...')


