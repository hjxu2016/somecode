#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:39:59 2017

@author: hjxu
"""

# coding=utf-8
import sys
sys.path.append('/home/hjxu/caffe-master/caffe/python')
import caffe
import numpy as np

root = '/home/hjxu/caffe_examples/metastatic/Alexnet/'
deploy =root + 'profile/deploy.prototxt'
caffe_model = '/home/hjxu/caffe_examples/metastatic/Alexnet/Alexnet__iter_150000.caffemodel'  # 训练好的 caffemodel
img = '/home/hjxu_disk/train-1/normal-fp/Normal_001/tumor_700000.png'  # 随机找的一张待测图片
#labels_filename = root + 'profile/labels.txt'  # 类别名称文件，将数字标签转换回类别名称
mean_file = '/home/hjxu/WSI-metastic/mean.npy'   #加载均值文件
# mean_file1 = mean_file.array[(1,1),(227,227)]
net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network

# 图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,257,257)
transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(257,257,3)变为(3,257,257)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    # 减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

im = caffe.io.load_image(img)  # 加载图片
net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

# 执行测试
out = net.forward()

# labels = np.loadtxt(labels_filename, str, delimiter='\t')  # 读取类别名称文件
prob = net.blobs['prob'].data[0].flatten()  # 取出最后一层（Softmax）属于某个类别的概率值，并打印
print prob
order = prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号
# print 'the class is:', labels[order]  # 将该序号转换成对应的类别名称，并打印
print 'the probability is:',float(prob[0])
print 'the class is:', order