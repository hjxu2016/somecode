clc;
clear all;
% tepisDIR is the directory which holds the MATLAB client for the tEPIS  
% This matlab client should be downloaded from:
% https://github.com/mitkovetta/tepis/tree/master/tepismat
tepisDIR = '/home/hjxu/MatlabProjects/tepis-master/tepismat';
addpath(genpath(tepisDIR));

openslide_include_path = '/home/hjxu/MatlabProjects/openslide-matlab-master';
% OpenSlide.initialize(openslide_include_path);
addpath('/home/hjxu/MatlabProjects/openslide-matlab-master');

% Directory where all the output .csv files are stored. The csv file names
% should be same as the input image filename.
result_dir = '/home/hjxu/breast_project/Evaluation_Matlab_hjxu/results';

% Directory in which the ground truth masks are stored
masks_dir = '/home/hjxu/breast_project/Evaluation_Matlab_hjxu/masks';

EVALUATION_MASK_LEVEL =5;
L0_RESOLUTION = 0.243;

[total_FPs, total_sensitivity, FP_summary, detection_summary] = generateFROC(result_dir, masks_dir, EVALUATION_MASK_LEVEL, L0_RESOLUTION);
plotFROC(total_FPs, total_sensitivity);