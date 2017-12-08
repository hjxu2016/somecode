clc;
clear all;
mask_org = imread('/home/hjxu/breast_project/ccf.png');
mask_gray = rgb2gray(mask_org);
mask = im2bw(mask_gray);
figure,imshow(mask_orgd);