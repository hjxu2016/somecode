#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:36:37 2017

@author: hjxu
"""
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

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def read_wsi_tumor(wsi_path):
    try:
        wsi_image = OpenSlide(wsi_path)
        m, n = wsi_image.dimensions
        level_used = wsi_image.level_count - 1
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                   wsi_image.level_dimensions[level_used]))
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

    return rgb_image

TUMOR_WSI_PATH = '/media/hjxu/Elements/Camelyon16/Training/Ground_Truth/Ground_Truth/Mask'
img_mask_save_dir = '/media/hjxu/Elements/Camelyon16_Result/Mask_resize'

wsi_paths = glob.glob(os.path.join(TUMOR_WSI_PATH, '*.tif'))
wsi_paths.sort()
WSI_path = list(wsi_paths)
i = 1
start=time.time()
for WSI in WSI_path:
     rgb_image = read_wsi_tumor(WSI)
     rgb_image_array = np.array(rgb_image)
     print ('No.%d' %i)
     print('%s Maskresize  in progress' % WSI)
     img_save_dir = img_mask_save_dir+'/'+ get_filename_from_path(WSI)
     scipy.misc.imsave(img_save_dir+'_mask_resize.tif', rgb_image_array)
     i = i + 1
     end = time.time()
     print ('run time%s'%(end-start))
print('has done...')
     
     