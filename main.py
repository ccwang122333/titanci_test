# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:59:41 2018

@author: cc_privide
"""
import datapre_function as dataf
filepath='D:/ccwang20160302/python/titanic_test/data/test.csv'
xtest=dataf.data_pre(filepath)
import pandas as pd
datatrain=pd.read_csv('D:/ccwang20160302/python/titanic_test/firsttest/data_pre.csv')
xtrain=datatrain.iloc[:,1:-1]
ytrain=datatrain.iloc[:,-1]

import sys
sys.path.append('D:/ccwang20160302/python/mlearn_program_cc/python')
import function_basis as fb
from imblearn.over_sampling import RandomOverSampler as ROS

ros_mod=ROS(random_state=555)
xtos,ytos=ros_mod.fit_sample(xtrain,ytrain)
ypre,yprob=fb.RFC_pack(xtos,xtest,ytos)

data_orign=pd.read_csv(filepath)
fare_null=data_orign[data_orign['Fare'].isnull()]
ytest_gender=pd.read_csv('D:/ccwang20160302/python/titanic_test/data/gender_submission.csv')
ytest=ytest_gender.drop([152])
ytest=ytest.reset_index()
del ytest['index']
n_err=0
n=len(ytest)
for i in range(n):
    if ytest['Survived'][i]!=ypre[i]:
        n_err+=1
res_acc=1-round(n_err/n,4)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
def roc_plot(fpr,tpr):
    plt.figure()
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    
fpr,tpr,thrd=roc_curve(ytest['Survived'],yprob[:,1],drop_intermediate=False)
auc_x=auc(fpr,tpr)
roc_plot(fpr,tpr)

    