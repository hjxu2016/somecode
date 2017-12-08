#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:58:55 2017

@author: hjxu
"""

#! coding=utf-8
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
import glob
import os

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
        (m,n) = wsi_image.dimensions
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
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>255

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    bounding_boxes, rgb_contour = get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


start=time.time()
root = '/home/hjxu/caffe_examples/metastatic/Alexnet/'
deploy =root + 'profile/deploy.prototxt'

model = '/home/hjxu/caffe_examples/metastatic/Alexnet/Alexnet__iter_150000.caffemodel'
model1 = '/media/hjxu/Elements/Camelyon16_Result/camelyon_model/alex_iter_186000.caffemodel'
mean='/home/hjxu_disk/lmdb/train_mean.npy'
mean1='/home/hjxu_disk/lmdb_10/train_mean.npy'


TUMOR_WSI_PATH = '//media/hjxu/My_Passport/hjxu_junzong/15chu'
patch_heat_save_dir = '/media/hjxu/My_Passport/hjxu_junzong/15chupredict'
#img_rgb_save_dir = '/home/hjxu_disk/Camelyon16_Result/train_result/normal_rgb'

net = caffe.Net(deploy, model, caffe.TEST)
net1 = caffe.Net(deploy,model1,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))



transformer1 = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer1.set_transpose('data', (2, 0, 1))
transformer1.set_mean('data', np.load(mean1).mean(1).mean(1))
transformer1.set_raw_scale('data', 255)
transformer1.set_channel_swap('data', (2, 1, 0))

wsi_paths = glob.glob(os.path.join(TUMOR_WSI_PATH, '*.ndpi'))
wsi_paths.sort()
WSI_path = list(wsi_paths)
i=1
for WSI in WSI_path:
        wsi_image, rgb_image,level,m,n=read_wsi_tumor(WSI)
        # bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
        # bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
        image_heat_save = np.zeros((n,m))

#        img_save_dir = img_rgb_save_dir+'/'+ get_filename_from_path(WSI)
        heat_save_dir = patch_heat_save_dir+'/'+ get_filename_from_path(WSI)
        print ('No.%d' %i)
        print('%s Classification is in progress' % WSI)
        # for bounding_box in bounding_boxes:
        #     b_x_start = int(bounding_box[0])
        #     b_y_start = int(bounding_box[1])
        #     b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        #     b_y_end = int(bounding_box[1]) + int(bounding_box[3])
        #     #        X = np.random.random_integers(b_x_start, high=b_x_end, size=500 )
        #     #        Y = np.random.random_integers(b_y_start, high=b_y_end, size=int((b_y_end-b_y_start)//2+1 ))
        #     col_cords = np.arange(b_x_start, b_x_end)
        #     row_cords = np.arange(b_y_start, b_y_end)
        mag_factor = 256
            #        for x, y in zip(X, Y):
                    #            if int(tumor_gt_mask[y, x]) != 0:
        for x in range (1,m) :
            for y in range (1,n):
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (256, 256))
                    img_tmp = skimage.img_as_float(np.array(patch))
                    img1 = np.tile(img_tmp, (1, 1, 3))
                    img2 = img1[:, :, :3]
                    net.blobs['data'].data[...] = transformer.preprocess('data', img2)
                    out = net.forward()
                    prob = out['prob'][0][0]
                    if prob > 0.5:
                        net1.blobs['data'].data[...] = transformer1.preprocess('data', img2)
                        out1 = net1.forward()
                        prob1 = out1['prob'][0][0]
                        if prob1 > 0.9:
                            prob = prob1
                        else:
                            prob = 0
                    image_heat_save[y, x] = prob#        np.save("/home/hjxu/Camelyon/Data/Tumor_110_new.npy",image_heat_save)
#        plt.imshow(image_heat_save)
#        plt.show()
#        scipy.misc.imsave(img_save_dir+'_rgb.tif', rgb_image)
        scipy.misc.imsave(heat_save_dir+'_predict.tif', image_heat_save)

        end = time.time()
        i = i+1
        print ('run time%s'%(end-start))
end1 = time.time()
print ('run time%s'%(end1-start))
print('has done...')

