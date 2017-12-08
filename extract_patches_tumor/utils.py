#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:22:46 2017

@author: hjxu
"""

PATCH_SIZE = 256
NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX = 100
NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500
TUMOR_WSI_PATH = '/media/hjxu/Elements/Camelyon16/Testset/Images_whole'
TUMOR_MASK_PATH = '/media/hjxu/Elements/Camelyon16/Testset/Ground_Truth/Masks'
PATCHES_TRAIN_POSITIVE_PATH = '/media/hjxu/Elements/Camelyon16_patch/test-0'
PATCH_TUMOR_PREFIX = 'tumor_'

patch_save_dir = PATCHES_TRAIN_POSITIVE_PATH
patch_prefix = PATCH_TUMOR_PREFIX
patch_index = 700000

#返回path的值\yanjiushengxujun\Extract_Patches_heatmap\wsi_mask_patch\Tumor_110_mask
def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename

#9.2号新增
def get_lastname_from_filepath(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    lenfile = len(filename)
    return filename[lenfile-14:lenfile]
