# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:44:26 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import datetime

#data_train=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)
#data_train.shape
##data_train=data_train.drop('is_trade',axis=1)
##data_train.shape
#data_test=pd.read_csv('D:/mother/round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)
#data_test.shape
#
##########删除训练数据的重复项
#data_train.drop_duplicates(inplace=True)
######给测试数据加上is_trade
##string2=data_train.columns.values.tolist()
##print(string2)
##string3=data_test.columns.values.tolist()
##print(string3)
#
#data_test['is_trade']=10
#
#
#def data_origin(data) :
#    ####令商品的category——list为a12
#    a1=data.item_category_list.map(lambda x:x.split(';'))
#    a11=a1.apply(lambda x:x[0])
#    a12=a1.apply(lambda x:x[1])
#    a12.value_counts()
#    data.item_category_list=a12
#            
#       ######广告商品属性列表数值化
#    b1=data.item_property_list.apply(lambda x:x.split(';'))
#    data['item_property_list_m1']=b1.apply(lambda x:x[0])
#    data['item_property_list_m2']=b1.apply(lambda x:x[1])
#    data['item_property_list_m3']=b1.apply(lambda x:x[2])
#    del data['item_property_list']
#    
#        #########时间戳时间格式化
#    c2=data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
#    c3=c2.astype(str).apply(lambda x:x.split(' '))
#    data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
#    data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))
#    #del data['context_timestamp']
#    del data['context_id']
#    
#    ####查询词预测类目属性表数值化
#    c1=data['predict_category_property'].apply(lambda x:x.split(';'))
#    c11=c1.apply(lambda x:x[0].split(':')[0])
#    data['predict_category_property_m1']=c1.apply(lambda x:x[0].split(':')[0])
#    del data['predict_category_property']
#    del data['instance_id']
#    data.item_category_list=data.item_category_list.apply(lambda x:int(x))
#    data.item_property_list_m1=data.item_property_list_m1.apply(lambda x:int(x))
#    data.item_property_list_m2=data.item_property_list_m2.apply(lambda x:int(x))
#    data.item_property_list_m3=data.item_property_list_m3.apply(lambda x:int(x))
#    data.predict_category_property_m1=data.predict_category_property_m1.apply(lambda x:int(x))
#    
#    return data
#
#
############合并数据
#data_train_test=pd.concat([data_train,data_test],ignore_index=True)
#data1=data_origin(data_train_test)
#
#
#data_y_train=data1[['day','is_trade']]
#
#data=data1.drop(['is_trade'],axis=1)
#
#string1=data.columns.values.tolist()
















new_train_data=pd.read_csv('light_data.csv')

string1=new_train_data.columns.values.tolist()

data=new_train_data.copy()

del data['user_id']
del data['item_id']

##########get_dummies
##############itrem_id有10236钟，item_brand_i有2075中
#string_cat=['item_id','item_category_list','item_brand_id','item_city_id','user_gender_id','user_age_level','user_occupation_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1']

#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_age_level','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1','item_property_list_m2', 'item_property_list_m3','hour','predict_category_property_m1']

string_cat=['item_category_list','item_brand_id','item_city_id','user_gender_id','user_age_level','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1','item_property_list_m2', 'item_property_list_m3','hour','predict_category_property_m1']


###########类别特征进行on-hot    
for i in range(len(string_cat)):
    locals()['df_'+string_cat[i]]=pd.get_dummies(data[string_cat[i]],prefix=string_cat[i])

category=pd.concat([df_item_category_list,df_item_city_id,df_user_gender_id,df_user_age_level,df_user_occupation_id,df_item_property_list_m1,df_item_property_list_m2,df_item_property_list_m3,df_hour,df_predict_category_property_m1],axis=1)

#######连续性特征
#
#continue_data=data.copy()
#string2=continue_data.columns.values.tolist()
#print(string2)
#
#del continue_data['user_id']
#del continue_data['item_id']
#del continue_data['item_brand_id']
#del continue_data['context_timestamp']
#del continue_data['shop_id']
#
#string2=continue_data.columns.values.tolist()
#print(string2)
#
#for i in range(len(string_cat)):
#    del continue_data[string_cat[i]]
#
#string2=continue_data.columns.values.tolist()
#print(string2)
############合并类别特征和连续性特征
#merge_data=pd.concat([category,continue_data],axis=1)


##########  
#
#all_item_data=pd.read_csv('all_item_data.csv')
#all_user_data=pd.read_csv('all_user_data.csv')
#all_page_data=pd.read_csv('all_page_data.csv')
#all_shop_data=pd.read_csv('all_shop_data.csv')
#all_two_data=pd.read_csv('all_two_data.csv')
#all_time_data=pd.read_csv('all_time_data.csv')
#
#new_data=data
#new_data=pd.merge(new_data,all_item_data,on=string1,how='left')
#new_data=pd.merge(new_data,all_user_data,on=string1,how='left')
#new_data=pd.merge(new_data,all_page_data,on=string1,how='left')
#new_data=pd.merge(new_data,all_shop_data,on=string1,how='left')
#new_data=pd.merge(new_data,all_two_data,on=string1,how='left')
#new_data=pd.merge(new_data,all_time_data,on=string1,how='left')

new_data=new_train_data.copy()

string2=new_data.columns.values.tolist()
print(string2)

#del new_data['context_timestamp']
######删除类别特征，保留数值型特征
string_new=['item_id','item_category_list','item_brand_id','item_city_id','user_gender_id','user_id','user_age_level','user_occupation_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1',]

for i in range(len(string_new)):
    del new_data[string_new[i]]

string2=new_data.columns.values.tolist()
##########合并one-hot和数值型特征
merge_data=pd.concat([category, new_data],axis=1)

string2=merge_data.columns.values.tolist()


new_train_data1=merge_data.copy()

#del new_train_data['user_brand_shop_see']
#del new_train_data['user_shop_see']
#del new_train_data['user_item_brand_see']
#del new_train_data['user_page_see']
#del new_train_data['context_timestamp']

#del new_train_data['user_id']


############进行18-24号数据的训
train=new_train_data1[(new_train_data1.day<24)].drop(['day','is_trade'],axis=1,inplace=False)
test=new_train_data1[(new_train_data1.day==24)].drop(['day','is_trade'],axis=1,inplace=False)
y_train=new_train_data1[(new_train_data1.day<24)][['is_trade']]
y_test=new_train_data1[(new_train_data1.day==24)][['is_trade']]


y_train =y_train.iloc[:,0].values
y_test =y_test.iloc[:,0].values


string2=train.columns.values.tolist()
print(string2)
 
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from sklearn.metrics import log_loss
import xgboost as xgb

def modelMetrics(clf,train_x,train_y,isCv=True,cv_folds=5,early_stopping_rounds=50):  
    if isCv:  
        xgb_param = clf.get_xgb_params()  
        xgtrain = xgb.DMatrix(train_x,label=train_y)  
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=clf.get_params()['n_estimators'],nfold=cv_folds,  
                          metrics='auc',early_stopping_rounds=early_stopping_rounds)#是否显示目前几颗树额  
        clf.set_params(n_estimators=cvresult.shape[0])  
  
    clf.fit(train_x,train_y,eval_metric='auc')  
  
    #预测  
    train_predictions = clf.predict(train_x)  
    train_predprob = clf.predict_proba(train_x)[:,1]#1的概率  
  
    #打印  
    print("\nModel Report")  
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))  
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))  
  
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)  
    feat_imp.plot(kind='bar',title='Feature importance')  
    plt.ylabel('Feature Importance Score')  



dtrain1=xgb.DMatrix(train,y_train)
dtest=xgb.DMatrix(test,y_test)


params={
        'booster':'gbtree',
        'objective':'binary:logistic',
        'eval_metric':'logloss',
#        'gamma':0.1
        'eta':0.05,
        'max_depth':3,
        'lambda':1,
        'subsample':0.7,
        'min_child_weight':1,
        'colsample_bytree':1,
        'colsample_bylevel':0.8,
        'seed':0,
        'n_thread':12
        
        }

import time
start=time.clock()

watchlist=[(dtrain1,'train'),(dtest,'val')]
mmodel=xgb.train(params,dtrain1,num_boost_round=10000,evals=watchlist,early_stopping_rounds=200)
ss=mmodel.best_iteration

elapsed=(time.clock()-start)
print("time used;",elapsed)















clf = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
nthread=4,# cpu 线程数 默认最大
learning_rate= 0.1, # 如同学习率
min_child_weight=1, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=3, # 构建树的深度，越大越容易过拟合
#gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=0.8, # 随机采样训练样本 训练实例的子采样比
#max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样 
#reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#reg_alpha=0, # L1 正则项参数
scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
objective= 'binary:logistic', #多分类的问题 指定学习任务和相应的学习目标

n_estimators=2000, #树的个数
seed=2018 #随机种子
#eval_metric= 'auc'
)  

#modelMetrics(clf,train,y_train) 
#clf.fit(train,y_train,eval_metric='logloss')
#设置验证集合 verbose=False不打印过程
clf.fit(train, y_train,eval_set=[(train, y_train), (test, y_test)],eval_metric='logloss',verbose=True)



#获取验证集合结果
evals_result = clf.evals_result()
y_true, y_pred = y_test, clf.predict_proba(test)[:,1]
#print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
print(log_loss(y_test,y_pred))  


clf.fit(train,y_train)
y_pred1 = clf.predict_proba(test)[:,1]


bao_cun=data_test[['instance_id']]
bao_cun['predicted_score']=y_pred1
bao_cun.to_csv('zgw6.txt',index=False,sep=' ')




from sklearn.model_selection import GridSearchCV
tuned_parameters= [{'n_estimators':[100,200,500],
                  'max_depth':[3,5,7], ##range(3,10,2)
                  'learning_rate':[0.5, 1.0],
                  'subsample':[0.75,0.8,0.85,0.9]
                  }]
tuned_parameters= [{'n_estimators':[100,200,500,1000]
                  }]
clf = GridSearchCV(clf, param_grid=tuned_parameters,scoring='roc_auc',n_jobs=4,iid=False,cv=5)  

clf.fit(X_train, y_train)
##clf.grid_scores_, clf.best_params_, clf.best_score_
print(clf.best_params_)
y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred)) 
y_proba=clf.predict_proba(X_test)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_true, y_proba))

from sklearn.model_selection import GridSearchCV
parameters= [{'learning_rate':[0.01,0.1,0.3],'n_estimators':[1000,1200,1500,2000,2500]}]
clf = GridSearchCV(XGBClassifier(
             max_depth=3,
             min_child_weight=1,
             gamma=0.5,
             subsample=0.6,
             colsample_bytree=0.6,
             objective= 'binary:logistic', #逻辑回归损失函数
             scale_pos_weight=1,
             reg_alpha=0,
             reg_lambda=1,
             seed=27
            ), 
            param_grid=parameters,scoring='roc_auc')  
clf.fit(X_train, y_train)
print(clf.best_params_)  
y_pre= clf.predict(X_test)
y_pro= clf.predict_proba(X_test)[:,1] 
print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro))
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre))

feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



