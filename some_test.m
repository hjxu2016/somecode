clc;
clear;
image = imread('/home/hjxu_disk/test.png');
target = imread('/home/hjxu_disk/patch_test001/tumor_700091.png');
image=stainnorm_reinhard(image,target);
figure,imshow(image);