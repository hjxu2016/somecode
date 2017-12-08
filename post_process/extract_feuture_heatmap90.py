#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:55:28 2017

@author: hjxu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:22:35 2017

@author: hjxu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:40:36 2017

@author: hjxu
"""
import csv
import glob
import os
import random
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label
from skimage.measure import regionprops

FILTER_DIM = 2
N_FEATURES = 31

MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4

heatmap_feature_names = ['name','region_count',  'longest_axis_largest_tumor',
                         'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                         'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                         'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                         'eccentricity_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                         'solidity_skew', 'solidity_kurt', 'label']

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def format_2f(number):
    return float("{0:.2f}".format(number))


def get_region_props(heatmap_threshold_2d, heatmap_prob_2d):
    labeled_img = label(heatmap_threshold_2d)
    return regionprops(labeled_img, intensity_image=heatmap_prob_2d)


def get_tumor_region_to_tissue_ratio(region_props, image_open):
    tissue_area = cv2.countNonZero(image_open)
    tumor_area = 0

    n_regions = len(region_props)
    for index in range(n_regions):
        tumor_area += region_props[index]['area']

    return float(tumor_area) / tissue_area

def get_largest_tumor_index(region_props):
    largest_tumor_index = -1

    largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index
            
    return largest_tumor_index


def get_longest_axis_in_largest_tumor_region(region_props, largest_tumor_region_index):
    largest_tumor_region = region_props[largest_tumor_region_index]
    return max(largest_tumor_region['major_axis_length'], largest_tumor_region['minor_axis_length'])


def get_average_prediction_across_tumor_regions(region_props):
    # close 255
    region_mean_intensity = [region.mean_intensity for region in region_props]
    return np.mean(region_mean_intensity)


def get_feature(region_props, n_region, feature_name):
    feature = [0] * 5
    if n_region > 0:
        feature_values = [region[feature_name] for region in region_props]
        feature[MAX] = format_2f(np.max(feature_values))
        feature[MEAN] = format_2f(np.mean(feature_values))
        feature[VARIANCE] = format_2f(np.var(feature_values))
        feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
        feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))

    return feature


def get_image_open(wsi_path):
    try:
        wsi_image = OpenSlide(wsi_path)
#        level_used = wsi_image.level_count - 1
        level_used=8    
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                   wsi_image.level_dimensions[level_used]))
        wsi_image.close()
    except OpenSlideUnsupportedFormatError:
        raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return image_open

    
def extract_features(image_name,heatmap_prob, image_open):
    """
        Feature list:
        -> (01) given t = 0.90, total number of tumor regions
        -> (02) given t = 0.90, percentage of tumor region over the whole tissue region
        -> (03) given t = 0.50, the area of largest tumor region
        -> (04) given t = 0.50, the longest axis in the largest tumor region
        -> (05) given t = 0.90, total number pixels with probability greater than 0.90
        -> (06) given t = 0.90, average prediction across tumor region
        -> (07-11) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'area'
        -> (12-16) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'perimeter'
        -> (17-21) given t = 0.90, max, mean, variance, skewness, and kurtosis of  'compactness(eccentricity[?])'
        -> (22-26) given t = 0.50, max, mean, variance, skewness, and kurtosis of  'rectangularity(extent)'
        -> (27-31) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'solidity'

    :param heatmap_prob:
    :param image_open:
    :return:

    """

    heatmap_threshold_t90 = np.array(heatmap_prob)

    heatmap_threshold_t90[heatmap_threshold_t90 < int(0.90 * 255)] = 0
    heatmap_threshold_t90[heatmap_threshold_t90 >= int(0.90 * 255)] = 255

    heatmap_threshold_t90_2d = np.reshape(heatmap_threshold_t90[:, :, :1],
                                          (heatmap_threshold_t90.shape[0], heatmap_threshold_t90.shape[1]))

    heatmap_prob_2d = np.reshape(heatmap_prob[:, :, :1],
                                 (heatmap_prob.shape[0], heatmap_prob.shape[1]))

    region_props_t90 = get_region_props(np.array(heatmap_threshold_t90_2d), heatmap_prob_2d)

    features = []
    features.append(image_name)
    f_count_tumor_region = len(region_props_t90)
    if f_count_tumor_region == 0:
        return [0.00] * N_FEATURES

    features.append(format_2f(f_count_tumor_region))

    f_percentage_tumor_over_tissue_region = get_tumor_region_to_tissue_ratio(region_props_t90, image_open)
    features.append(format_2f(f_percentage_tumor_over_tissue_region))

    largest_tumor_region_index_t90 = get_largest_tumor_index(region_props_t90)

    f_pixels_count_prob_gt_90 = cv2.countNonZero(heatmap_threshold_t90_2d)
    features.append(format_2f(f_pixels_count_prob_gt_90))

    f_avg_prediction_across_tumor_regions = get_average_prediction_across_tumor_regions(region_props_t90)
    features.append(format_2f(f_avg_prediction_across_tumor_regions))

    f_area = get_feature(region_props_t90, f_count_tumor_region, 'area')
    features += f_area

    f_perimeter = get_feature(region_props_t90, f_count_tumor_region, 'perimeter')
    features += f_perimeter

    f_eccentricity = get_feature(region_props_t90, f_count_tumor_region, 'eccentricity')
    features += f_eccentricity

    f_solidity = get_feature(region_props_t90, f_count_tumor_region, 'solidity')
    features += f_solidity

    return features

TRAIN_WSI_PATH = '/media/hjxu/Elements/Camelyon16/Testset/Images_whole'
train_wsi_paths = glob.glob(os.path.join(TRAIN_WSI_PATH, '*.tif'))
train_wsi_paths.sort()
pridict_path = '/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict_sample/'

f_test = '/home/hjxu/breast_project/post_process/features/feature_test.csv'

features_file_test = open(f_test, 'w')

wr_test = csv.writer(features_file_test, quoting=csv.QUOTE_NONNUMERIC)
wr_test.writerow(heatmap_feature_names[:len(heatmap_feature_names) - 1])
i=1


for wsi_path in train_wsi_paths:
    print i
    i = i+1
    wsi_name = get_filename_from_path(wsi_path)
    image_open = get_image_open(wsi_path)
    
    heatmap_prob_path = pridict_path + wsi_name+'_predict.tif'
    heatmap_prob = cv2.imread(heatmap_prob_path)
    features = extract_features(wsi_name,heatmap_prob, image_open)
    
    
############训练的时候需要label，需要取消注释############
############测试的时候不需要label，需要注释##################
#    if 'umor' in wsi_name:
#        features += [1]
#    else:
#        features += [0]
###########测试的时候不需要label，需要注释##################
############测试的时候不需要label，需要注释##################
#    print(features)
     
    features = []