#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:26:44 2017

@author: hjxu
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle
from sklearn import svm

HEATMAP_FEATURE_CSV_TRAIN = '/home/hjxu/breast_project/post_process/features/feature_train.csv'
model_save_pickle = '/home/hjxu/breast_project/post_process/tree/Random_model_90.pkl'

df_train = pd.read_csv(HEATMAP_FEATURE_CSV_TRAIN)

#df_validation = pd.read_csv(utils.HEATMAP_FEATURE_CSV_VALIDATION)

n_columns = len(df_train.columns)
feature_column_names = df_train.columns[1:n_columns - 1]#返回的是每个列表的第一行，对应的是特征的名字
#print feature_column_names
label_column_name = df_train.columns[n_columns - 1]
print label_column_name

train_x = df_train[feature_column_names]
train_y = df_train[label_column_name]

###############c测试文件##################3
df_test = pd.read_csv('/home/hjxu/breast_project/post_process/features/feature_test.csv')
df_testy = pd.read_csv('/home/hjxu/breast_project/Extract_Features_heatmap/GT_ground.csv')
test_x = df_test[feature_column_names]
test_y = df_testy['label']
###############c测试文件##################3
clf = svm.SVC()
#clf = RandomForestClassifier(n_estimators=500, n_jobs=2)  #分类型决策树
s = clf.fit(train_x, train_y) # 训练模型


#with open(model_save_pickle, "wb") as f:
#     pickle.dump(clf, f)


prob_predict_y_validation = clf.predict_proba(test_x)#给出带有概率值的结果，每个点所有label的概率和为1
predict = clf.predict(test_x)
#for i in range (130):
#    print df_test['name'][i],prob_predict_y_validation[i],predict[i]
#######测试模型的准确率##########
#r = clf.score(test_x,test_y)
#print r
########测试模型的准确率##########3




#############画出roc曲线###################
#predictions_validation = prob_predict_y_validation[:, 1]
#fpr, tpr, _ = roc_curve(test_y, predictions_validation)
#
#roc_auc = auc(fpr, tpr)
#plt.title('ROC Validation')
#plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0, 1], [0, 1], 'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
###################画roc曲线##############