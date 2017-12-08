#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:50:35 2017

@author: hjxu
"""
import openslide
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from scipy import misc
import scipy

import time
import matplotlib.pyplot as plt


from PIL import Image
filename = '/home/hjxu_disk/Camelyon/Tumor_72-110/tumor_091.tif'
slide = openslide.OpenSlide(filename)
level_count = slide.level_count 
print 'level_count = ',level_count 
