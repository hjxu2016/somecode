#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:27:29 2017

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
import os
import scipy
import cv2
import skimage
from skimage import measure,morphology
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
caffe.set_mode_gpu()
caffe.set_device(0)

from xml.dom import minidom
def create_coordinate(Annotations,example):
    Annotation = doc.createElement("Annotation")
    Annotation.setAttribute("Name",example["Name"])
    Annotation.setAttribute("Type",example["Type"])
    Annotation.setAttribute("Prob", example["Prob"])
    Annotation.setAttribute("PartOfGroup","None")
    Annotation.setAttribute("Color", example["Color"])
    Annotations.appendChild(Annotation)
    Coordinates = doc.createElement("Coordinates")
    # Coordinates.setAttribute("Text",example["Text"])
    Annotation.appendChild(Coordinates)
    Coordinate1 = doc.createElement("Coordinate")
    Coordinate1.setAttribute("Order",example["Order1"])
    Coordinate1.setAttribute("X", example["X1"])
    Coordinate1.setAttribute("Y", example["Y1"])
    Coordinates.appendChild(Coordinate1)
    Coordinate2 = doc.createElement("Coordinate")
    Coordinate2.setAttribute("Order", example["Order2"])
    Coordinate2.setAttribute("X", example["X2"])
    Coordinate2.setAttribute("Y", example["Y2"])
    Coordinates.appendChild(Coordinate2)
    Coordinate3 = doc.createElement("Coordinate")
    Coordinate3.setAttribute("Order", example["Order3"])
    Coordinate3.setAttribute("X", example["X3"])
    Coordinate3.setAttribute("Y", example["Y3"])
    Coordinates.appendChild(Coordinate3)
    Coordinate4 = doc.createElement("Coordinate")
    Coordinate4.setAttribute("Order", example["Order4"])
    Coordinate4.setAttribute("X", example["X4"])
    Coordinate4.setAttribute("Y", example["Y4"])
    Coordinates.appendChild(Coordinate4)
    return Coordinates



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
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>0

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    # plt.imshow(image_open)
    # plt.show()
    bounding_boxes, rgb_contour = get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open


start=time.time()
root = '/home/hjxu/caffe_examples/metastatic/Alexnet/'
deploy =root + 'profile/deploy.prototxt'
model = '/home/hjxu/caffe_examples/metastatic/Alexnet/Alexnet__iter_150000.caffemodel'
model1 = '/media/hjxu/Elements/Camelyon16_Result/camelyon_model/alex_iter_186000.caffemodel'
mean='/home/hjxu_disk/lmdb/train_mean.npy'
mean1='/home/hjxu_disk/lmdb_10/train_mean.npy'

WSI = '/home/hjxu_disk/Camelyon/Tumor/Tumor_026.tif'
save_XML = '/home/hjxu/breast_project/reports/'
wsi_image, rgb_image,level,m,n=read_wsi_tumor(WSI)
rows, cols, channels = rgb_image.shape
# rgb_image = cv2.resize(rgb_image, (cols/2, rows/2), interpolation=cv2.INTER_AREA)
bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
image_heat_save = np.zeros((n, m))

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

doc = minidom.Document()
ASAP_Annotation = doc.createElement("ASAP_Annotations")

doc.appendChild(ASAP_Annotation)
ASAP_Annotation.setAttribute("Type", "HIS")
Annotations = doc.createElement("Annotations")
ASAP_Annotation.appendChild(Annotations)
AnnotationGroups = doc.createElement("AnnotationGroups")
ASAP_Annotation.appendChild(AnnotationGroups)


print('%s Classification is in progress' % WSI)
mag_factor = 256
num_cells=0
count=0
for bounding_box in bounding_boxes:
    b_x_start = int(bounding_box[0])
    b_y_start = int(bounding_box[1])
    b_x_end = int(bounding_box[0]) + int(bounding_box[2])
    b_y_end = int(bounding_box[1]) + int(bounding_box[3])
    #        X = np.random.random_integers(b_x_start, high=b_x_end, size=500 )
    #        Y = np.random.random_integers(b_y_start, high=b_y_end, size=int((b_y_end-b_y_start)//2+1 ))
    col_cords = np.arange(b_x_start, b_x_end)
    row_cords = np.arange(b_y_start, b_y_end)
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
#                img3 = img2
                # patch.save('/home/hjxu/caffe_examples/metastatic/Alexnet/temp/temp.png', 'PNG')
                # tile = tile = np.asarray(Image.open('/home/hjxu/caffe_examples/metastatic/Alexnet/temp/temp.png'))
                # img = tile[:, :, 0:3]
                # scipy.misc.imsave('/home/hjxu/caffe_examples/metastatic/Alexnet/temp1.png', img)
                # img2 = caffe.io.load_image('/home/hjxu/caffe_examples/metastatic/Alexnet/temp1.png')
                net.blobs['data'].data[...] = transformer.preprocess('data', img2)
                out = net.forward()
                prob = out['prob'][0][0]
                WSI_x1 =  x_large
                WSI_y1 =  y_large
                WSI_x2 =  x_large + 256
                WSI_y2 =  y_large
                WSI_x3 =  x_large + 256
                WSI_y3 =  y_large + 256
                WSI_x4 =  x_large
                WSI_y4 =  y_large + 256
                if prob>0.5:
                    net1.blobs['data'].data[...] = transformer1.preprocess('data', img2)
                    out1 = net1.forward()
                    prob1 = out1['prob'][0][0]
                    if prob1>0.5:
                        prob = prob1
                        d = {}
                        d["Id"] = str(count)
                        d["Name"] = "Annotation" + " " + str(count)
                        count = count + 1
                        d["Color"] = "#F4FA58"
                        d["Prob"] = str(prob)
                        # d["Text"]=str(output_prob[output_prob.argmax()])
                        d["Type"] = "Polygon"
                        d["Order1"] = "0"
                        d["X1"] = str(WSI_x1)
                        d["Y1"] = str(WSI_y1)
                        d["Order2"] = "1"
                        d["X2"] = str(WSI_x2)
                        d["Y2"] = str(WSI_y2)
                        d["Order3"] = "2"
                        d["X3"] = str(WSI_x3)
                        d["Y3"] = str(WSI_y3)
                        d["Order4"] = "3"
                        d["X4"] = str(WSI_x4)
                        d["Y4"] = str(WSI_y4)
                        create_coordinate(Annotations, d)
                        count=count+1
                    else:
                        prob = 0
                image_heat_save[y, x] = prob
                
xml_name = save_XML + os.path.splitext(WSI)[0].split('/')[-1] + '.xml'
f = file(xml_name, "w")
doc.writexml(f)
f.close()
del doc
# np.save("/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict2/Tumor_110.npy", image_heat_save)
plt.imshow(image_heat_save)
plt.show()
scipy.misc.imsave('/home/hjxu/breast_project/reports/026.tif', image_heat_save)
end = time.time()
print ('run time%s'%(end-start))
print('has done...')
