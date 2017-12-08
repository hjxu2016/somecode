clc
clear all;
MaskPath = '/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict_sample/Test_004_predict.tif';
SavePath_xml = '/home/hjxu/breast_project/reports/Test_004_predict.xml';
Multiple = 256;% Multiple is upsamples
mask2xml(MaskPath,SavePath_xml,Multiple);
disp('has done...')