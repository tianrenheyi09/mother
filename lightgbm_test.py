# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:21:06 2018

@author: 1
"""



import pandas as pd
import numpy as np

################凭借set1,2,3



dataset1=pd.read_csv('data_set1.csv')
dataset2=pd.read_csv('data_set2.csv')
dataset3=pd.read_csv('data_set3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset12=pd.concat([dataset1,dataset2],axis=0)

del dataset1['context_timestamp']
del dataset2['context_timestamp']
del dataset12['context_timestamp']

train=dataset1.drop(['is_trade','day'],axis=1,inplace=False)
test=dataset2.drop(['is_trade','day'],axis=1,inplace=False)
y_train=dataset1.is_trade
y_test=dataset2.is_trade

#new_train_data=dataset1.copy()
#
##del new_train_data['user_brand_shop_see']
##del new_train_data['user_shop_see']
##del new_train_data['user_item_brand_see']
##del new_train_data['user_page_see']
#del new_train_data['context_timestamp']
#
##del new_train_data['user_id']
#
#train=new_train_data[(new_train_data.day<24)].drop(['is_trade','day'],axis=1,inplace=False)
#test=new_train_data[(new_train_data.day==24)].drop(['is_trade','day'],axis=1,inplace=False)
#y_train=data_y_train[(data_y_train.day<24)][['is_trade']]
#y_test=data_y_train[(data_y_train.day==24)][['is_trade']]



##############---------------lightGBM转换数据格式--------------------
import json
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import log_loss


print("load data")
X_train=train
#y_train =y_train.iloc[:,0].values
#y_test =y_test.iloc[:,0].values

y_train =y_train.values
y_test =y_test.values

X_test = test
string2=train.columns.values.tolist()
print(string2)              
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id']

string3=['item_id','item_category_list','item_brand_id','item_city_id','user_gender_id','user_id','user_age_level','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1','item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1',
         'is_new_user_brand','is_new_user_item_cate','is_new_user_shop']

###########-------------LIGHTgbm-------------------
lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=31,
        max_depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
#         min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
lgb_model = lgb0.fit(X_train, y_train, eval_set=[(X_test,y_test)],eval_metric='logloss',feature_name=string2,categorical_feature=string3,early_stopping_rounds=200,verbose=True)
best_iter = lgb_model.best_iteration

#######24号的误差-
pred1= lgb_model.predict_proba(test)[:, 1]
print(log_loss(y_test,pred1))     

############进行18-24号数据的训练
train=new_train_data[(new_train_data.day<25)].drop(['day','is_trade'],axis=1,inplace=False)
test=new_train_data[(new_train_data.day==25)].drop(['day','is_trade'],axis=1,inplace=False)
y_train=data_y_train[data_y_train.day<25][['is_trade']]
y_test=data_y_train[data_y_train.day==25][['is_trade']]

print("load data")
X_train=train
y_train =y_train.iloc[:,0].values
y_test =y_test.iloc[:,0].values
X_test = test
string2=train.columns.values.tolist()
print(string2)              
string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_age_level','user_occupation_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1',]



lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=31,
        max_depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=best_iter)

lgb_model = lgb0.fit(X_train, y_train,feature_name=string2,categorical_feature=string3)
pred2= lgb_model.predict_proba(X_test)[:, 1]

bao_cun=data_test[['instance_id']]
bao_cun['predicted_score']=pred2
bao_cun.to_csv('zgw5.txt',index=False,sep=' ')

