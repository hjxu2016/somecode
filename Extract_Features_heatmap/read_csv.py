#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:39:11 2017

@author: hjxu
"""




import csv

csv_file = '/home/hjxu/breast_project/Extract_Features_heatmap/GT.csv'
#with open(csv_file, "r") as f:
#    reader = csv.reader(f)
#    column1 = [row[0] for row in reader]
#print column1   

with open(csv_file, "r") as f:
    reader = csv.reader(f)
    row1 = [row for row in reader]
print row1[2] 
#print reader

#with open(csv_file, "r") as f:
#    reader = csv.reader(f)
#    column0 = [row[0] for row in reader]
#print(column0[2])

#csv_file_label = '/home/hjxu/breast_project/Extract_Features_heatmap/GT_ground.csv'
#with open(csv_file_label,"w") as csvfile: 
#    writer = csv.writer( csvfile)
#    writer.writerow(['name','label'])
#    for i in range(len(column0)):
#        if column1[i] == 'Tumor':
#            label = 1;
#        else:
#            label = 0
#        writer.writerow([column0[i],label])
#print (len(column))
#
#for i in range (len(column)):
    
    
