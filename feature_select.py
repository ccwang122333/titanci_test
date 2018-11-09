# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:20:35 2018

@author: cc_privide
"""
import pandas as pd
import numpy as np
data=pd.read_csv('D:/ccwang20160302/python/titanic_test/firsttest/data_pre.csv')
#data_orign=pd.read_csv('D:/ccwang20160302/python/titanic_test/data/train.csv')
xdata=np.array(data.iloc[:,1:-1])
n_col=np.shape(xdata)[1]
ydata=np.array(data.iloc[:,-1])

import sys
sys.path.append('D:/ccwang20160302/python/mlearn_program_cc/python')
import function_basis as fb

#fs=fb.feature_value_PS(xdata,ydata)

from sklearn.model_selection import StratifiedKFold as sfk
from imblearn.over_sampling import RandomOverSampler as ROS
#from scipy.stats import pearsonr
sfk_model=sfk(n_splits=10,random_state=10)
temp2=[]
res_all=[]
for itrain,itest in sfk_model.split(xdata,ydata):
    xtrain=xdata[itrain]
    ytrain=ydata[itrain]
    xtest=xdata[itest]
    ytest=ydata[itest]
    ros_mod=ROS(random_state=555)
    xtos,ytos=ros_mod.fit_sample(xtrain,ytrain)
    ypre=fb.RFC_pack(xtrain,xtest,ytrain)
    nerror=0
    nsum=len(ytest)
    for i in range(nsum):
        if ypre[i]!=ytest[i]:
            nerror+=1
    temp2.append(round(nerror*100/nsum,2))
    res_x=fb.result_ml(ytest,ypre)
    res_all.append(res_x)