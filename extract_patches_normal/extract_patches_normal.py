#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:01:00 2017

@author: hjxu
"""
import glob
import numpy as np
import cv2
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import utils
import os
import matplotlib.pyplot as plt
import scipy
from scipy import misc



def get_bbox_normal(cont_img, image):
    rgb_contour = image.copy()
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_color = (255, 0, 0)  # blue color code
    cv2.drawContours(rgb_contour, contours, -1, line_color, 2)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    return bounding_boxes, rgb_contour

def read_wsi_normal(wsi_path):
    """
        # =====================================================================================
        # read WSI image and resize
        # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
        # ======================================================================================
    """
    try:
        wsi_image = OpenSlide(wsi_path)
#        level_used = wsi_image.level_count - 1
        level_used = 8
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                   wsi_image.level_dimensions[level_used]))

    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None

    return wsi_image, rgb_image, level_used


def find_roi_bbox_normal( rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        bounding_boxes, rgb_contour = get_bbox_normal(image_open, rgb_image)
        return bounding_boxes, image_open



def extract_negative_patches_from_normal_wsi(wsi_image, image_open, level_used,
                                             bounding_boxes, patch_save_dir, patch_prefix,
                                             patch_index):
    """
        Extract negative patches from Normal WSIs

        Save extracted patches to desk as .png image files

        :param wsi_image:
        :param image_open:
        :param level_used:
        :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
        :param patch_save_dir: directory to save patches into
        :param patch_prefix: prefix for patch name
        :param patch_index:
        :return:

    """

    mag_factor = pow(2, level_used)

    print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        b_y_end = int(bounding_box[1]) + int(bounding_box[3])
        X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
        Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

        for x, y in zip(X, Y):
            if int(image_open[y, x]) == 1:
                x_large = x * mag_factor
                y_large = y * mag_factor
                patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                scipy.misc.imsave(patch_save_dir + patch_prefix + str(patch_index)+'.png', np.array(patch)[:,:,:3])   
                patch_index += 1
                patch.close()

    return patch_index

wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
wsi_paths.sort()

#image_mask_pair = zip(wsi_paths)
image_mask_pair = list(wsi_paths)

for image_path in image_mask_pair:
    print('extract_negative_patches_from_tumor_wsi(): %s' % utils.get_filename_from_path(image_path))
    save_dir =utils.patch_save_dir+'/'+ utils.get_filename_from_path(image_path)+'/'
    print('save_dir is():%s' %save_dir)
    if (os.path.exists(save_dir)):
        print(' has create,we will skip this file' )
        print('This file is %s' % utils.get_lastname_from_filepath(image_path))
    else:
        os.makedirs(save_dir)
        
        wsi_image, rgb_image, level_used =  read_wsi_normal(image_path)
        
        #tumor_gt_mask =  cv2.imread(TUMOR_PATCH)
        
        bounding_boxes,image_open = find_roi_bbox_normal( rgb_image)
        
        patch_index = extract_negative_patches_from_normal_wsi(wsi_image, np.array(image_open),
                                                                                 level_used, bounding_boxes,
                                                                                 save_dir, utils.patch_prefix,
                                                                                 utils.patch_index)
        wsi_image.close()
