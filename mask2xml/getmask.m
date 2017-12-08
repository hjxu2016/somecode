clc
clear all;
mask = imread('/home/hjxu_disk/Camelyon16_Result/train_result/Mask_predict1_1-70/Tumor_001_heat.tif');
mask1 = im2bw(mask,0.9);
figure,imshow(mask1);
[m,n] = find(mask1==1);
[a,~] = size(m);