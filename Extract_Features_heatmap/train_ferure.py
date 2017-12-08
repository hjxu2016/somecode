#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:56:54 2017

@author: hjxu
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle

df_validation = pd.read_csv('/home/hjxu/Camelyon/camelyon16-grand-challenge-master-hjux-study/camelyon16/postprocess/features/heatmap_features_validation.csv')
n_columns = len(df_validation.columns)
feature_column_names = df_validation.columns[:n_columns - 1]#返回的是每个列表的第一行，对应的是特征的名字
label_column_name = df_validation.columns[n_columns - 1]

df_train = pd.read_csv('/home/hjxu/DuktoReceived/feature_train1.csv')
train_x = df_train[feature_column_names]
train_y = df_train[label_column_name]

clf = RandomForestClassifier(n_estimators=50, n_jobs=2)  #分类型决策树
s = clf.fit(train_x, train_y)
#with open("/home/hjxu/breast_project/Extract_Features_heatmap/train_Rest.pkl", "wb") as f:
#    pickle.dump(clf, f)


df_test = pd.read_csv('/home/hjxu/Camelyon/Data/features/features.csv')
df_testy = pd.read_csv('/home/hjxu/breast_project/Extract_Features_heatmap/GT_ground.csv')
test_x = df_test[feature_column_names]
#test_y = df_testy['label']
prob_predict_y_validation = clf.predict_proba(test_x)
#r = clf.score(test_x,test_y)

print(prob_predict_y_validation)