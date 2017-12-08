clc;
clear all;
mask1 = imread('/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict1/Test_002_predict.tif');
mask = mask1;
mask2 = imread('/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict2/test_002.tif');
[m,n] = size(mask1);
% mask3 = im2bw(mask1,0.9);
for x = 1:m
    for y = 1:n
        if (mask1(x,y)>=0.5*255 & mask2(x,y)<0.5*255)
            mask(x,y) =0;
        end
    end
end
            
figure,imshow(mask1);
figure,imshow(mask);
% figure,imshow(mask2)
imwrite(mask,'/home/hjxu_disk/Camelyon16_Result/test_result/Mask_predict_sample/test_002.tif');
    