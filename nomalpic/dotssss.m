
clc;clear all;close all;
addpath('/home/hjxu/breast_project/nomalpic/unit/');
image_folder='/home/hjxu/breast_project/nomalpic/org_img/';
savepath = '/home/hjxu/breast_project/nomalpic/img_result/';
image_ext = '.png';
folder_content = dir ([image_folder,'*',image_ext]);
n_folder_content = size (folder_content,1); 
target=imread('/home/hjxu/breast_project/nomalpic/tumor_700038.png');
for k= 1: length(folder_content)
    t0 = clock;
    fprintf(' %d/%d to be Operated...\n ', k,n_folder_content);  
    string_img = [image_folder,folder_content(k,1).name];
    name=folder_content(k,1).name;
    ind=strfind(name,'.png');
    name=name(1:ind(1)-1);
    image = imread(string_img); 
    image=stainnorm_reinhard(image,target);
    img=image(:,:,[1 2 3]);
    imwrite(img,[savepath name '.png']);
end

