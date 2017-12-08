# coding=utf-8
caffe_root = '/home/hjxu/caffe-master/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import numpy as np
import openslide
import time
import os
import skimage.io
import math
import glob

################xml生成#######################################
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


##################分类########################################
def cell_classifition(img_path):
    cls_net.blobs['data'].data[...] = transformer.preprocess('data', img_path)
    output = cls_net.forward()
    output_prob = output['prob'][0]
    print output_prob
    print 'predicted class is:', output_prob.argmax() + 1  # 1 2 3
    cls = output_prob.argmax() + 1
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    print 'output label:', labels[output_prob.argmax()]

############################### 图片 #####################################################
MRXS_Path='/media/hjxu/My_Passport/hjxu_junzong/To_lb/'
mrxs_list=glob.glob(MRXS_Path+'/*.ndpi')
save_XML='/media/hjxu/My_Passport/hjxu_junzong/To_lb_xml/'
############################### 分类网络 #################################################
profile_root='/home/hjxu/caffe_examples/metastatic/Alexnet/'
cls_deploy=profile_root+'profile/deploy.prototxt'
cls_model=profile_root+'Alexnet__iter_150000.caffemodel'
mean_proto_path='/home/hjxu/WSI-metastic/mean.binaryproto'
labels_file = '/home/hjxu/breast_project/lb_zc_xml/label.txt'
cls_net=caffe.Net(cls_deploy,cls_model,caffe.TEST)
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_proto_path, 'rb' ).read()
blob.ParseFromString(data)
array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
transformer = caffe.io.Transformer({'data': cls_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mean_npy.mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
#########################################################################################
for WSI in mrxs_list:
    doc = minidom.Document()
    ASAP_Annotation = doc.createElement("ASAP_Annotations")

    doc.appendChild(ASAP_Annotation)
    ASAP_Annotation.setAttribute("Type", "HIS")
    Annotations = doc.createElement("Annotations")
    ASAP_Annotation.appendChild(Annotations)
    AnnotationGroups = doc.createElement("AnnotationGroups")
    ASAP_Annotation.appendChild(AnnotationGroups)
    print WSI
    slide=openslide.open_slide(WSI)
    OVslide=slide.level_dimensions[0]#Overview
    [width, height]=OVslide
    seg_w=256#segnet输入尺寸
    seg_h=256

    w_times=int(math.floor(2*(width-seg_w)/seg_w+1))#行滑动次数
    h_times=int(math.floor(2*(height-seg_h)/seg_h+1))#列滑动次数

    outImg_w=(w_times-1)*200+seg_w#输出图像宽
    outImg_h=(h_times-1)*200+seg_h#输出图像高
    num_cells=0
    count=0
    for x in range(0,outImg_w,256):
        for y in range(0,outImg_h,256):
            # X=seg_w*x/2
            # Y=seg_h*y/2
            img_slide_block = np.array(slide.read_region((x,y), 0, (256, 256)))
            save_img = skimage.img_as_float(img_slide_block).astype(np.float32)
            cls_net.blobs['data'].data[...] = transformer.preprocess('data', save_img)
            output = cls_net.forward()
            output_prob = output['prob'][0]
            print 'predicted class is:', output_prob.argmax() + 1
            cls = output_prob.argmax() + 1
            labels = np.loadtxt(labels_file, str, delimiter='\t')
            print 'output label:', labels[output_prob.argmax()]
            prob=output_prob[0]
            print '预测为转移的概率:',prob
            WSI_x1 =  x
            WSI_y1 =  y
            WSI_x2 =  x + 256
            WSI_y2 =  y
            WSI_x3 =  x + 256
            WSI_y3 =  y + 256
            WSI_x4 =  x
            WSI_y4 =  y + 256
            if cls==1:
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
    xml_name = save_XML + os.path.splitext(WSI)[0].split('/')[-1] + '.xml'
    f = file(xml_name, "w")
    doc.writexml(f)
    f.close()
    del doc

end = time.time()
print ('The NDPI END!!!!')


