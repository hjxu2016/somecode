#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:22:46 2017

@author: hjxu
import os,shutil,string
"""
import scipy
import glob
import os
from openslide import OpenSlide
import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt
from PIL import Image


#def apply_lut(tile, lut):
#    """ Apply look-up-table to tile to normalize H&E staining. """
#    ps = tile.shape  # tile size is (rows, cols, channels)
#    reshaped_tile = tile.reshape((ps[0] * ps[1], 3))
#    normalized_tile = np.zeros((ps[0] * ps[1], 3))
#    idxs = range(ps[0] * ps[1])
#    Index = 256 * 256 * reshaped_tile[idxs, 0] + 256 * reshaped_tile[idxs, 1] + reshaped_tile[idxs, 2]
#    normalized_tile[idxs] = lut[Index.astype(int)]
#    return normalized_tile.reshape(ps[0], ps[1], 3).astype(np.uint8)
#
#file_path = '/media/hjxu/Elements/Camelyon16_patch/train/train-3-save-0/label-0-0/Tumor_001/tumor_700000.png'
#lut_file_path = '/home/hjxu/PycharmProjects/rename/tumor_700000.png'
#tile = np.asarray(Image.open(file_path))
#tile = tile[:, :, 0:3]
## tile.save('/home/hjxu/PycharmProjects/rename/tu_normalized.tif')
#print tile.shape
##
#lut = np.asarray(Image.open(lut_file_path)).squeeze()
#lut = lut[:, :, :3]
#tile_normalized = apply_lut(tile, lut)
## im = Image.fromarray(tile_normalized.astype(np.uint8))
## im.save('/home/hjxu/PycharmProjects/rename/tu_normalized.tif')
#plt.imshow(tile_normalized)
#plt.show()

mask = cv2.imread('/home/hjxu_disk/Camelyon16_Result/train_result/normal_predict1_all/Normal_009_predict.tif');
tumor_gt_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)