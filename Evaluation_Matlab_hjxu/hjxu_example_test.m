clc;
clear all;
% detection
%
% @author: Babak Ehteshami Bejnordi
%
%
% Input:
% -------------------------------------
% result_dir:	Directory containing the CSV files;
% masks_dir:	Directory containing the groundtruth Masks;
% EVALUATION_MASK_LEVEL:  The level at which the evaluation mask is made;
% L0_RESOLUTION:	Pixel resolution of the image at level 0;
%
% Output:
% -------------------------------------
% total_FPs:    an array containing the average number of false positives 
%         per image for different thresholds;
%
% total_sensitivity: an array containing overall sensitivity of the system 
%         for different thresholds;
%
% detection_summary:   A matlab cell array with detection details for each 
%        image. Each row in the cell represent the detection details of a 
%        single image and contains a matlab structure array with 'fields' 
%        that are the labels of the lesions that should be detected 
%        (non-ITC tumors) and 'values' that contain detection details 
%        [confidence score, X-coordinate, Y-coordinate]. Lesions that are 
%        missed by the algorithm have an empty value.
%
% FP_summary:   A matlab cell array which lists the false positive detections  
%        for each image. Each row in the cell represent the detection details 
%        of a single image and contains a matlab structure array with 'fields'   
%        that represent the false positive finding number and 'values' that   
%        contain detection details [confidence score, X-coordinate, Y-coordinate]. 
% -------------------------------------
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

CSV_file_list = dir(fullfile(result_dir,'*.csv'));
CSV_file_list = {CSV_file_list.name}';

FROC_data = cell(size(CSV_file_list,1),4);
FP_summary = cell(size(CSV_file_list,1),2);
detection_summary = cell(size(CSV_file_list,1),2);

for i=1:numel(CSV_file_list)
    caseID = CSV_file_list{i}(1:end-4);
    fprintf('Evaluating Performance on image: %s\n',caseID);
%     is_tumor = strcmp(CSV_file_list{i}(1:5),'Tumor');
is_tumor = 1
    fid = fopen(fullfile(result_dir, CSV_file_list{i}));
    results = textscan(fid,'%f %d32 %d32','Delimiter',',');
    fclose(fid);
%      results = imread('/home/hjxu/breast_project/Evaluation_Matlab_hjxu/results/Test_001.tif')   ;
    if (is_tumor)
        mask_fname = fullfile(masks_dir, strcat(CSV_file_list{i}(1:end-4),'_Mask.tif'));     %# full path to file
        slide = openslide_open(mask_fname);
        %this is my
        [width, height] = openslide_get_level_dimensions(slide , EVALUATION_MASK_LEVEL + 1);
        original_mask=  openslide_read_region(slide,0,0,width,height,'level',EVALUATION_MASK_LEVEL + 1);
%         original_mask = slide(:, :, EVALUATION_MASK_LEVEL + 1);
        evaluation_mask = computeEvaluationMask(original_mask, EVALUATION_MASK_LEVEL, L0_RESOLUTION);
        Isolated_Tumor_Cells = computeITCList(evaluation_mask, EVALUATION_MASK_LEVEL, L0_RESOLUTION);
    else
        Isolated_Tumor_Cells = [];
        evaluation_mask = 0;
    end
    
    FROC_data{i, 1} = caseID;
    FP_summary{i, 1} = caseID;
    detection_summary{i, 1} = caseID;
    [FROC_data{i, 2}, FROC_data{i, 3}, FROC_data{i, 4}, FP_summary{i,2}, detection_summary{i,2}] ...
        = compute_FP_TP_Probs(results, 1, evaluation_mask, Isolated_Tumor_Cells, EVALUATION_MASK_LEVEL);  

end

[total_FPs, total_sensitivity] = computeFROC(FROC_data);
plotFROC(total_FPs, total_sensitivity);