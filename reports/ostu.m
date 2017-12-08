clc;
clear all;
img = imread('/media/hjxu/Elements/Camelyon16_Result/tumor_pre_for_ROC/tumor_085_predict.tif');
img_bw = im2bw(img,0.9);
figure,imshow(img_bw);
imwrite(img_bw,'/home/hjxu/breast_project/reports/tumor_085_0.9.tif');