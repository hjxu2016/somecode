#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:56:26 2017

@author: hjxu
"""
import numpy as np
from PIL import Image
#def more(A,B):
#    N = len(A)
#    for i in range (0,N):
#        if A[i]>B[i]:
#            C[i]=1
#        else :
#            C[i]=0
#    return C
#        
def RGB2Lab(R,G,B):
#    %RGB2LAB Convert an image from RGB to CIELAB
#%
#% function [L, a, b] = RGB2Lab(R, G, B)
#% function [L, a, b] = RGB2Lab(I)
#% function I = RGB2Lab(...)
#%
#% RGB2Lab takes red, green, and blue matrices, or a single M x N x 3 image, 
#% and returns an image in the CIELAB color space.  RGB values can be
#% either between 0 and 1 or between 0 and 255.  Values for L are in the
#% range [0,100] while a and b are roughly in the range [-110,110].  The
#% output is of type double.
#%
#% This transform is based on ITU-R Recommendation BT.709 using the D65
#% white point reference. The error in transforming RGB -> Lab -> RGB is
#% approximately 10^-5.  
#%
#% See also LAB2RGB.
#
#% By Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
#% Updated for MATLAB 5 28 January 1998.
#% Updated for MATLAB 7 30 March 2009.
    R = np.array(R)
    G = np.array(G)
    B = np.array(B)
    if np.max(np.max(R)) > 1.0 and np.max(np.max(G)) > 1.0 and np.max(np.max(B)) > 1.0:
          R = (R) / 255
          G = (G) / 255
          B = (B) / 255
          
    T = 0.008856 #Set a threshold
    [M, N] = R.shape
    s = M * N
    RGB = np.zeros((s,3))
    RGB[:,0] = R.reshape((1,s))
    RGB[:,1] = G.reshape((1,s))
    RGB[:,2] = B.reshape((1,s))
#    RGB = np.array([R.reshape((1,s)),G.reshape((1,s)),B.reshape((1,s))])
    MAT = [[0.412453,0.357580,0.180423],
           [0.212671,0.715160,0.072169],
           [0.019334,0.119193,0.950227]]
    
    XYZ = np.dot(RGB , MAT)
#    Normalize for D65 white point
    X = XYZ[1,:] / 0.950456
    Y = XYZ[2,:]
    Z = XYZ[3,:] / 1.088754
    
    XT = X > T
    YT = Y > T
    ZT = Z > T
    Y3 = Y**(1/3)

    fX = XT*X**(1/3)+(~XT)*(7.787 * X + 16/116)

    fY = YT * Y3 + (~YT) * (7.787 * Y + 16/116)

    fZ = ZT * Z**(1/3) + (~ZT) * (7.787 * Z + 16/116)
    
    L = (YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y)).reshape( (M, N))
    a = (500 * (fX - fY)).reshape((M, N))
    b =( 200 * (fY - fZ)).reshape(( M, N))
    
#    L = [[L],[a],[b]]
    return L,a,b

def Lab2RGB(L,a,b):
#    %LAB2RGB Convert an image from CIELAB to RGB
#%
#% function [R, G, B] = Lab2RGB(L, a, b)
#% function [R, G, B] = Lab2RGB(I)
#% function I = Lab2RGB(...)
#%
#% Lab2RGB takes L, a, and b double matrices, or an M x N x 3 double
#% image, and returns an image in the RGB color space.  Values for L are in
#% the range [0,100] while a* and b* are roughly in the range [-110,110].
#% If 3 outputs are specified, the values will be returned as doubles in the
#% range [0,1], otherwise the values will be uint8s in the range [0,255].
#%
#% This transform is based on ITU-R Recommendation BT.709 using the D65
#% white point reference. The error in transforming RGB -> Lab -> RGB is
#% approximately 10^-5.  
#%
#% See also RGB2LAB. 
#
#% By Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
#% Updated for MATLAB 5 28 January 1998.
#% Fixed a bug in conversion back to uint8 9 September 1999.
#% Updated for MATLAB 7 30 March 2009.
    T1 = 0.008856
    T2 = 0.206893
    [M, N] = L.shape
    s = M * N
    L = L.reshape((1,s))
    a = a.reshape((1,s))
    b = b.reshape((1,s))
#    % Compute Y
    fY = ((L + 16) / 116) ** 3
    YT = fY > T1
    fY = (~YT) * (L / 903.3) + YT * fY
    Y = fY
    
#    % Alter fY slightly for further calculations
    fY = YT * (fY ** (1/3)) + (~YT) * (7.787 * fY + 16/116)  

#    % Compute X
    fX = a / 500 + fY
    XT = fX > T2
    X = (XT * (fX ** 3) + (~XT) * ((fX - 16/116) / 7.787))
    
#    % Compute Z
    fZ = fY - b / 200
    ZT = fZ > T2
    Z = (ZT * (fZ ** 3) + (~ZT) * ((fZ - 16/116) / 7.787))
    
#    % Normalize for D65 white point
    X = X * 0.950456
    Z = Z * 1.088754
    
#    % XYZ to RGB
    MAT = [ [3.240479,-1.537150,-0.498535],
           [-0.969256  ,1.875992  ,0.041556],
           [0.055648,-0.204043 ,1.057311]]
    
    RGB = max(min(MAT * [X ,Y,Z], 1), 0)
        
    R = RGB[1,:].reshape(( M, N))
    G = RGB[2,:].reshape((M, N))
    B = RGB[3,:].reshape(( M, N))
    
    return R,G,B

def stainnorm_reinhard(source,target):
    source = np.array(source)
    target = np.array(target)
    [x ,y ,z] = np.array(source).shape
    [x1 ,y1 ,z1] = np.array(target).shape
    source1=np.zeros((x,y,3))
    target1=np.zeros((x1,y1,3))
#    source=float(source)
#    target=float(target)
    
    src1,src2,  src3=RGB2Lab(source[:,:,0],source[:,:,1],source[:,:,2])
    
    
    
    src_1=src1.reshape((1,x*y))
    src_1 = float(src_1)
    
    src_2=src2.reshape((1,x*y))
    src_2=float(src_2)
    
    src_3=src3.reshape((1,x*y))
    src_3=float(src_3)
    
    std1=np.std(src_1,1)         # %Finding out standard deviation of individual 
    std2=np.std(src_2,1)         #%channels of source image
    std3=np.std(src_3,1)
    
    m1=np.mean(np.mean(source1[:,:,1])) #%Finding out mean of individual channels 
    m2=np.mean(np.mean(source1[:,:,2])) #%of source image
    m3=np.mean(np.mean(source1[:,:,3]))
    
    tgt1,tgt2,tgt3=RGB2Lab(target)
    
    tgt_1=tgt1.reshape((1,x1*y1))
    tgt_1=float(tgt_1)

    tgt_2=tgt2.reshape((1,x1*y1))
    tgt_2=float(tgt_2)
    
    tgt_3=tgt3.reshape((1,x1*y1))
    tgt_3=float(tgt_3)

    std4=np.std(tgt_1,1)      #    %Finding out standard deviation of individual 
    std5=np.std(tgt_2,1)       #   %channels of target image
    std6=np.std(tgt_3,1)
    m4=np.mean(np.mean(target1[:,:,1])) #%Finding out mean of individual channels 
    m5=np.mean(np.mean(target1[:,:,2])) #%target image
    m6=np.mean(np.mean(target1[:,:,3]))

    result=np.zeros((x,y,3))
    
    for i in range (1,x):
        for j in range (1,y):
            result[i,j,1]=((src1-m1)*(std4/std1))+m4
            result[i,j,2]=((src2-m2)*(std5/std2))+m5
            result[i,j,3]=((src3-m3)*(std6/std3))+m6
    results = np.zeros(x,y,3)       
    norm_img1,norm_img2,norm_img3=Lab2RGB(result[i,j,1],result[i,j,2],result[i,j,3])
    results[:,:,1] = norm_img1
    results[:,:,2] = norm_img2
    results[:,:,3] = norm_img3
    return results

image = Image.open('/home/hjxu/breast_project/nomalpic_python/Tumor_001_1_9_0_1.png')
target = Image.open('/home/hjxu/breast_project/nomalpic_python/tumor_700040.png')
image=stainnorm_reinhard(image,target)


    

