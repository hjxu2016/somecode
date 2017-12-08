#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:17:14 2017

@author: hjxu
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle
#'/home/hjxu/Camelyon/Data/features/features.csv'
df_validation = pd.read_csv('/home/hjxu/Camelyon/camelyon16-grand-challenge-master-hjux-study/camelyon16/postprocess/features/heatmap_features_validation.csv')
n_columns = len(df_validation.columns)
feature_column_names = df_validation.columns[:n_columns - 1]#返回的是每个列表的第一行，对应的是特征的名字
label_column_name = df_validation.columns[n_columns - 1]
validation_x = df_validation[feature_column_names]
validation_y = df_validation[label_column_name]
#print feature_column_names
df_test = pd.read_csv('/home/hjxu/breast_project/Extract_Features_heatmap/feature_test.csv')
df_testy = pd.read_csv('/home/hjxu/breast_project/Extract_Features_heatmap/GT_ground.csv')
test_x = df_test[feature_column_names]
test_y = df_testy['label']
#print test_x.shape
with open("/home/hjxu/breast_project/Extract_Features_heatmap/train_Rest.pkl", "rb") as f:
    clf = pickle.load(f)
##    

#predict_y_validation = clf.predict(test_x)#直接给出预测结果，每个点在所有label的概率和为1，内部还是调用predict——proba()
#r = clf.score(test_x,test_y)
#
#print(r)




prob_predict_y_validation = clf.predict_proba(test_x)#给出带有概率值的结果，每个点所有label的概率和为1


#############画出roc曲线###################
predictions_validation = prob_predict_y_validation[:, 1]
fpr, tpr, _ = roc_curve(test_y, predictions_validation)

roc_auc = auc(fpr, tpr)
plt.title('ROC Validation')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
###################画roc曲线############################