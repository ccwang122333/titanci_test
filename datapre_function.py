# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:07:58 2018

@author: cc_privide
"""
import pandas as pd
import numpy as np
def data_pre(filepath):
    data=pd.read_csv(filepath)
    data=data[data['Embarked'].notnull()]
    data=data[data['Fare'].notnull()]
    data=data.reset_index(drop=True)#reset_index重置索引，否则数据按照原索引，删除的行索引跳过
    n_col=np.shape(data)[1]
    n_row=np.shape(data)[0]
    colnames=data.columns.values
    if 'Survived' in colnames:
        ydata=data['Survived']
    sex_dummies=pd.get_dummies(data['Sex'],prefix='sex')
    Em_dummies=pd.get_dummies(data['Embarked'],prefix='Em')
    #统计票号分为数字和字母两类,
    ticket_dig=[]
    ticket_str=[]
    for xi in data['Ticket']:
        if 48<=ord(xi[0])<=57:
            ticket_dig.append(1)
            ticket_str.append(0)
        else:
            ticket_dig.append(0)
            ticket_str.append(1)
    ticket_dig=pd.DataFrame(ticket_dig)
    ticket_str=pd.DataFrame(ticket_str)
    ticket_dig.columns=['ticket_dig']
    ticket_str.columns=['ticket_str']
    Pclass_dummies=pd.get_dummies(data['Pclass'],prefix='Pclass')
    #划分有无兄弟姐妹配偶两类
    SibSp_n=[]
    SibSp_y=[]
    for xi in data['SibSp']:
        if xi==0:
            SibSp_n.append(1)
            SibSp_y.append(0)
        else:
            SibSp_n.append(0)
            SibSp_y.append(1)
    SibSp_n=pd.DataFrame(SibSp_n)
    SibSp_y=pd.DataFrame(SibSp_y)
    SibSp_n.columns=['SibSp_n']
    SibSp_y.columns=['SibSp_y']
    #父母孩子采用同上的处理
    Parch_n=[]
    Parch_y=[]
    for xi in data['Parch']:
        if xi==0:
            Parch_n.append(1)
            Parch_y.append(0)
        else:
            Parch_n.append(0)
            Parch_y.append(1)
    Parch_n=pd.DataFrame(Parch_n)
    Parch_y=pd.DataFrame(Parch_y)
    Parch_n.columns=['Parch_n']
    Parch_y.columns=['Parch_y']
    from sklearn import preprocessing
    fare_pre=preprocessing.minmax_scale(data['Fare'])
    fare_pre=pd.DataFrame(fare_pre)
    fare_pre.columns=['fare_pre']
#    from collections import Counter
#    type_Cabin=Counter(data['Cabin'])
    #拟合缺失的cabin,采用随机森林分类
    from sklearn.ensemble import RandomForestClassifier as RFC
    data_cabin=pd.concat([Pclass_dummies,sex_dummies,Em_dummies,SibSp_n,SibSp_y,
                        Parch_n,Parch_y,ticket_dig,ticket_str,fare_pre,data['Cabin']],axis=1)
    #print(data_cabin.info())
    train_cabin=data_cabin[data_cabin['Cabin'].notnull()]
    test_cabin=data_cabin[data_cabin['Cabin'].isnull()]
    train_cabin=train_cabin.reset_index(drop=True)
    test_cabin=test_cabin.reset_index(drop=True)
    Cabin_type=[]
    for xi in train_cabin['Cabin']:
        if xi[0]=='A':
            Cabin_type.append(1)
        elif xi[0]=='B':
            Cabin_type.append(2)
        elif xi[0]=='C':
            Cabin_type.append(3)
        elif xi[0]=='D':
            Cabin_type.append(4)
        elif xi[0]=='E':
            Cabin_type.append(5)
        elif xi[0]=='F':
            Cabin_type.append(6)
        else:Cabin_type.append(0)
    del train_cabin['Cabin']#删除该列 
    Cabin_type=pd.DataFrame(Cabin_type)
    Cabin_type.columns=['Cabin_type']   
    train_cabin=pd.concat([train_cabin,Cabin_type],axis=1)        
    train_cabin=np.array(train_cabin)
    test_cabin=np.array(test_cabin)
    xtrain_cabin=train_cabin[:,:-1]        
    ytrain_cabin=train_cabin[:,-1]
    xtest_cabin=test_cabin[:,:-1]
    RFC_model=RFC(random_state=10)
    RFC_model.fit(xtrain_cabin,ytrain_cabin)
    pre_nullcabin=RFC_model.predict(xtest_cabin)
    new_cabin=np.hstack((ytrain_cabin,pre_nullcabin))
    new_cabin_df=pd.DataFrame(new_cabin)
    new_cabin_df.columns=['cabin_pre']
    #拟合缺失的age,用逻辑回归标签问题，最终使用随机森林回归
    from sklearn.ensemble import RandomForestRegressor as RFR
    #from sklearn.linear_model import LogisticRegression as LR
    data_age=pd.concat([Pclass_dummies,sex_dummies,Em_dummies,SibSp_n,SibSp_y,
                        Parch_n,Parch_y,ticket_dig,ticket_str,fare_pre,data['Age']],axis=1)
    data_nonull=np.array(pd.concat([Pclass_dummies,sex_dummies,Em_dummies,SibSp_n,SibSp_y,
                        Parch_n,Parch_y,ticket_dig,ticket_str,fare_pre],axis=1))
    train_age=data_age[data_age['Age'].notnull()]
    test_age=data_age[data_age['Age'].isnull()]
    train_age=np.array(train_age.reset_index(drop=True))
    test_age=np.array(test_age.reset_index(drop=True))
    xtrain_age=train_age[:,:-1]
    ytrain_age=train_age[:,-1]
    xtest_age=test_age[:,:-1]
    RFR_model=RFR(random_state=11)
    RFR_model.fit(xtrain_age,ytrain_age)
    pre_nullage=RFR_model.predict(xtest_age)
    pre_nullage=pre_nullage.round()#对所有数挨个取整
    new_age=np.hstack((ytrain_age,pre_nullage))#行拼接
    new_age_df=pd.DataFrame(new_age)
    new_age_df.columns=['age_pre']
    
    data_pre=pd.concat([Pclass_dummies,sex_dummies,Em_dummies,SibSp_n,SibSp_y,
                        Parch_n,Parch_y,ticket_dig,ticket_str,fare_pre,new_age_df,new_cabin_df],axis=1)
    return data_pre        