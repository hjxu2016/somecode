clc;
clear all;
img = imread('/home/hjxu/breast_project/reports/1027/tumor_026_lb.tif');
% img1 = cat(3,img,img,img);
% % img = double(img/255);
% % img = double(img);
% img2 = rgb2gray(img1);
figure,imshow(img);