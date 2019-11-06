# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:39:23 2018

@author: 1
"""


import numpy as np
import pandas as pd
import datetime

data=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)
data.shape

data1=data.copy()
 
    ####令商品的category——list为a12
a1=data.item_category_list.map(lambda x:x.split(';'))
a11=a1.apply(lambda x:x[0])
a12=a1.apply(lambda x:x[1])
a12.value_counts()
data.item_category_list=a12

        
   ######广告商品属性列表数值化
b1=data.item_property_list.apply(lambda x:x.split(';'))
data['item_property_list_m1']=b1.apply(lambda x:x[0])
data['item_property_list_m2']=b1.apply(lambda x:x[1])
data['item_property_list_m3']=b1.apply(lambda x:x[2])

del data['item_property_list']

    #########时间戳时间格式化
c2=data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
c3=c2.astype(str).apply(lambda x:x.split(' '))
data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))

del data['context_timestamp']

del data['context_id']



####查询词预测类目属性表数值化

c1=data['predict_category_property'].apply(lambda x:x.split(';'))
c11=c1.apply(lambda x:x[0].split(':')[0])
data['predict_category_property_m1']=c1.apply(lambda x:x[0].split(':')[0])


del data['predict_category_property']

del data['instance_id']


###############挖掘其他隐含特征
u=data[['item_id']]
u.drop_duplicates(inplace=True)

###商品浏览总次数
u1=data[['item_id']]
u1['item_is_see']=1
u1=u1.groupby(['item_id']).agg('sum').reset_index()

item_feature=pd.merge(u,u1,on=['item_id'],how='left')


######商品成交总次数
#u2=data[['item_id','is_trade']]
#u2=u2[(u2.is_trade==1)][['item_id']]
#
#u2['item_is_trade']=1
#u2=u2.groupby(['item_id']).agg('sum').reset_index()
#
#
#
#item_feature=pd.merge(item_feature,u2,on=['item_id'],how='left')
##
########商品成交率
#item_feature=item_feature.fillna(0)
#item_feature['item_%%trade']=item_feature.item_is_trade/item_feature.item_is_see


#####商品不同品牌浏览总数
u1=data[['item_brand_id']]
u1['item_brand_see']=1
u1=u1.groupby(['item_brand_id']).agg('sum').reset_index()

item_brand_feature=u1
#######商品不同品牌成交次数
#u2=data[(data.is_trade==1)][['item_brand_id']]
#u2['item_brand_trade']=1
#u2=u2.groupby(['item_brand_id']).agg('sum').reset_index()
#
#######s商品不同同品成交率
#item_brand_feature=pd.merge(u1,u2,on=['item_brand_id'],how='left')
#item_brand_feature=item_brand_feature.fillna(0)
#item_brand_feature['item_brand_%%trade']=item_brand_feature.item_brand_trade/item_brand_feature.item_brand_see


#####y用户浏览总次数
u1=data[['user_id']]
u1['user_id_see']=1
u1=u1.groupby('user_id').agg('sum').reset_index()

user_feature=u1

#####用户成交次数
#u2=data[(data.is_trade==1)][['user_id']]
#u2['user_trade']=1
#u2=u2.groupby('user_id').agg('sum').reset_index()
#
######用户历史成交率
#user_feature=pd.merge(u1,u2,on=['user_id'],how='left')
#user_feature=user_feature.fillna(0)
#user_feature['user_%%trade']=user_feature.user_trade/user_feature.user_id_see

####上下文page对应的浏览数和点击

u1=data[['context_page_id']]
u1['page_see']=1
u1=u1.groupby(['context_page_id']).agg('sum').reset_index()

page_feature=u1

#u2=data[(data.is_trade==1)][['context_page_id']]
#u2['page_trade']=1
#u2=u2.groupby(['context_page_id']).agg('sum').reset_index()
#
#
#page_feature=pd.merge(u1,u2,on=['context_page_id'],how='left')
#page_feature=page_feature.fillna(0)
#
#
#page_feature['page_%%trade']=page_feature.page_trade/page_feature.page_see

######店铺的浏览次数
#u1=data[['context_timestamp']]
#u1['context_timestamp_see']=1
#u1=u1.groupby('context_timestamp').agg('sum').reset_index()

u1=data[['shop_id']]
u1['shop_id_see']=1
u1=u1.groupby('shop_id').agg('sum').reset_index()

shop_feature=u1
#####店铺的成交次数
#u2=data[(data.is_trade==1)][['shop_id']]
#u2['shop_id_trade']=1
#u2=u2.groupby('shop_id').agg('sum').reset_index()
#####店铺的成交率
#shop_feature=pd.merge(u1,u2,on=['shop_id'],how='left')
#shop_feature=shop_feature.fillna(0)
#shop_feature['shop_%%trade']=shop_feature.shop_id_trade/shop_feature.shop_id_see

#####用户和商品编号的之间的特征
u1=data[['user_id','item_id']]
u1['user_item_see']=1
u1=u1.groupby(['user_id','item_id']).agg('sum').reset_index()

user_item_feature=u1

#u2=data[(data.is_trade==1)][['user_id','item_id']]
#u2['user_item_trade']=1
#u2=u2.groupby(['user_id','item_id']).agg('sum').reset_index()
#
#user_item_feature=pd.merge(u1,u2,on=['user_id','item_id'],how='left')
#user_item_feature=user_item_feature.fillna(0)

#########用户和商品 品牌之间的特征
u1=data[['user_id','item_brand_id']]
u1['user_item_brand_see']=1
u1=u1.groupby(['user_id','item_brand_id']).agg('sum').reset_index()

user_brand_feature=u1

#u2=data[(data.is_trade==1)][['user_id','item_brand_id']]
#u2['user_item_brand_trade']=1
#u2=u2.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
#
#user_brand_feature=pd.merge(u1,u2,on=['user_id','item_brand_id'],how='left')
#user_brand_feature=user_brand_feature.fillna(0)


##########用户和上下文时间的特征
#u1=data[['user_id','context_timestamp']]
#u1['user_time_see']=1
#u1=u1.groupby(['user_id','context_timestamp']).agg('sum').reset_index()
#
#u2=data[(data.is_trade==1)][['user_id','context_timestamp']]
#u2['user_time_trade']=1
#u2=u2.groupby(['user_id','context_timestamp']).agg('sum').reset_index()
#
#user_time_feature=pd.merge(u1,u2,on=['user_id','context_timestamp'],how='left')
#user_time_feature=user_time_feature.fillna(0)


###########用户和店铺的特征
u1=data[['user_id','shop_id']]
u1['user_shop_see']=1
u1=u1.groupby(['user_id','shop_id']).agg('sum').reset_index()

user_shop_feature=u1

#u2=data[(data.is_trade==1)][['user_id','shop_id']]
#u2['user_shop_trade']=1
#u2=u2.groupby(['user_id','shop_id']).agg('sum').reset_index()
#
#user_shop_feature=pd.merge(u1,u2,on=['user_id','shop_id'],how='left')
#user_shop_feature=user_shop_feature.fillna(0)


######用户和上下文page之间的联系
u1=data[['user_id','context_page_id']]
u1['user_page_see']=1
u1=u1.groupby(['user_id','context_page_id']).agg('sum').reset_index()

user_page_feature=u1


#u2=data[(data.is_trade==1)][['user_id','context_page_id']]
#u2['user_page_trade']=1
#u2=u2.groupby(['user_id','context_page_id']).agg('sum').reset_index()
#
#user_page_feature=pd.merge(u1,u2,on=['user_id','context_page_id'],how='left')
#user_page_feature=user_page_feature.fillna(0)


#############用户和店铺以及品牌的特征
u1=data[['user_id','item_brand_id','shop_id']]
u1['user_brand_shop_see']=1
u1=u1.groupby(['user_id','item_brand_id','shop_id']).agg('sum').reset_index()

user_brand_shop_feature=u1

#u2=data[data.is_trade==1][['user_id','item_brand_id','shop_id']]
#u2['user_brand_shop_trade']=1
#
#u2=u2.groupby(['user_id','item_brand_id','shop_id']).agg('sum').reset_index()
#
#user_brand_shop_feature=pd.merge(u1,u2,on=['user_id','item_brand_id','shop_id'],how='left')
#
#user_brand_shop_feature=user_brand_shop_feature.fillna(0)


######用户和商品品牌以及商品页码的特征
u1=data[['user_id','item_brand_id','context_page_id']]
u1['user_brand_page_see']=1
u1=u1.groupby(['user_id','item_brand_id','context_page_id']).agg('sum').reset_index()

user_brand_page_feature=u1


#u2=data[(data.is_trade==1)][['user_id','item_brand_id','context_page_id']]
#u2['user_brand_page_trade']=1
#u2=u2.groupby(['user_id','item_brand_id','context_page_id']).agg('sum').reset_index()
#
#user_brand_page_feature=pd.merge(u1,u2,on=['user_id','item_brand_id','context_page_id'],how='left')
#user_brand_page_feature=user_brand_page_feature.fillna(0)



########商品品牌编号和上下文广告商品展示编号的特征
u1=data[['item_brand_id','context_page_id']]
u1['brand_page_see']=1
u1=u1.groupby(['item_brand_id','context_page_id']).agg('sum').reset_index()

brand_page_feature=u1




#u2=data[(data.is_trade==1)][['item_brand_id','context_page_id']]
#u2['brand_page_trade']=1
#u2=u2.groupby(['item_brand_id','context_page_id']).agg('sum').reset_index()
#
#brand_page_feature=pd.merge(u1,u2,on=['item_brand_id','context_page_id'],how='left')
#brand_page_feature=brand_page_feature.fillna(0)

#######上下文展示编号和店铺id间的特征
u1=data[['context_page_id','shop_id']]
u1['page_shop_see']=1
u1=u1.groupby(['context_page_id','shop_id']).agg('sum').reset_index()

page_shop_feature=u1


#u2=data[(data.is_trade==1)][['context_page_id','shop_id']]
#u2['page_shop_trade']=1
#u2=u2.groupby(['context_page_id','shop_id']).agg('sum').reset_index()
#
#
#page_shop_feature=pd.merge(u1,u2,on=['context_page_id','shop_id'],how='left')
#page_shop_feature=page_shop_feature.fillna(0)



string2=data.columns.values.tolist()
print(string2)

###########合并另外提取的特征
new_feature_data=data


new_feature_data=pd.merge(new_feature_data,item_feature,on='item_id',how='left')
new_feature_data=pd.merge(new_feature_data,item_brand_feature,on='item_brand_id',how='left')
new_feature_data=pd.merge(new_feature_data,user_feature,on='user_id',how='left')
new_feature_data=pd.merge(new_feature_data,page_feature,on='context_page_id',how='left')
new_feature_data=pd.merge(new_feature_data,shop_feature,on='shop_id',how='left')
new_feature_data=pd.merge(new_feature_data,user_item_feature,on=['user_id','item_id'],how='left')


new_feature_data=pd.merge(new_feature_data,user_brand_feature,on=['user_id','item_brand_id'],how='left')
#new_feature_data=pd.merge(new_feature_data,user_time_feature,on=['user_id','context_timestamp'],how='left')
new_feature_data=pd.merge(new_feature_data,user_shop_feature,on=['user_id','shop_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_page_feature,on=['user_id','context_page_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_brand_shop_feature,on=['user_id','item_brand_id','shop_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_brand_page_feature,on=['user_id','item_brand_id','context_page_id'],how='left')
new_feature_data=pd.merge(new_feature_data,brand_page_feature,on=['item_brand_id','context_page_id'],how='left')
new_feature_data=pd.merge(new_feature_data,page_shop_feature,on=['context_page_id','shop_id'],how='left')


new_feature_data=new_feature_data.fillna(0)



#data=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',nrows=10000,delimiter=' ',header=0)
#data=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)
#data.shape
#
#data1=data.copy()


#new_feature_data.to_csv('new_feature_data.csv',index=None)

new_train_data=new_feature_data



new_train_data.item_category_list=new_train_data.item_category_list.apply(lambda x:int(x))
new_train_data.item_property_list_m1=new_train_data.item_property_list_m1.apply(lambda x:int(x))
new_train_data.item_property_list_m2=new_train_data.item_property_list_m2.apply(lambda x:int(x))
new_train_data.item_property_list_m3=new_train_data.item_property_list_m3.apply(lambda x:int(x))
new_train_data.predict_category_property_m1=new_train_data.predict_category_property_m1.apply(lambda x:int(x))

del new_train_data['item_city_id']

train=new_train_data[(new_train_data.day<24)].drop(['day','is_trade'],axis=1,inplace=False)
test=new_train_data[(new_train_data.day==24)].drop(['day','is_trade'],axis=1,inplace=False)
y_train=new_train_data[(new_train_data.day<24)][['is_trade']]
y_test=new_train_data[(new_train_data.day==24)][['is_trade']]




############xgboost预测

#import xgboost as xgb
#
#params={'booster':'gbtree',
#        'objective':'rank:pairwise',
#        'gamma':0.1,
#        'min_child_weight':1.1,
#        'max_depth':5,
#        'lambda':10,
#        'subsample':0.7,
#        'colsample_bytree':0.7,
#        'colsample_bylevel':0.7,
#        'eta':0.01,
#        'tree_method':'exact',
#        'seed':0,
#        'nthread':12
#                }
#
#
#x_train=xgb.DMatrix(train.values,label=y_train.values)
#x_test=xgb.DMatrix(test.values)
#
#model=xgb.train(params,x_train,600)
#
#predict=model.predict(x_test)
#
#from sklearn.metrics import log_loss
#
#print(log_loss(y_test,predict))




##############lightGBM进行预测
string2=train.columns.values.tolist()
print(string2)
string3=['item_id','item_category_list','item_brand_id','user_id','user_gender_id','user_age_level','user_occupation_id','shop_id','item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3', 'hour', 'predict_category_property_m1']

#
#features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
#                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
#                'user_age_level', 'user_star_level',
#                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level','context_page_id','item_city_id'
#                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
#                ]


import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV



from catboost import CatBoostClassifier
cat_features=[0,1,2,7,8,9,10,12,13,14,20,21,22,23,24]

model=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,loss_function='Logloss')
model.fit(train,y_train,cat_features)


preds_proba=model.predict_proba(test)[:,1]

print(log_loss(y_test,preds_proba))




#turn_params=[{'objective':['binary'],'learning_rate':[0.01,0.03,0.05,0.1],'n_estimators':[60,80,100],'max_depth':[6,7,8]}]
#
#clf=GridSearchCV(lgb.LGBMClassifier(seed=7),turn_params,scoring='roc_auc')
#clf.fit(train,y_train)
#print('best params of lgb is:',clf.best_params_)

clf=lgb.LGBMClassifier(num_leaves=30,max_depth=4,learning_rate=0.1,n_estimators=300)
clf.fit(train,y_train,feature_name=string2,categorical_feature=string3)


y_pre=clf.predict_proba(test)[:,1]

print(log_loss(y_test,y_pre))




##############进行全部数据的训练
train=new_train_data.drop(['is_trade'],axis=1,inplace=False)
y_train=new_train_data[['is_trade']]

string2=train.columns.values.tolist()
print(string2)
string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id']

import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV

clf=lgb.LGBMClassifier(num_leaves=50,max_depth=8,learning_rate=0.1,n_estimators=180)
clf.fit(train,y_train,feature_name=string2,categorical_feature=string3)


data_test=pd.read_csv('round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)

new_data_test=data_precoss(data_test)

y_pre_score=clf.predict_proba(new_data_test)[:,1]

bao_cun=data_test[['instance_id']]
bao_cun['predict_score']=y_pre_score

bao_cun.to_csv('zgw.txt',index=False,sep=' ')































######划分需要进行编码的特征数据

dataset1=new_feature_data.loc[:,['item_id','item_category_list','item_brand_id','item_city_id','user_id',
               'user_gender_id','user_occupation_id','context_timestamp','context_page_id','shop_id']]



#dataset2=new_feature_data.loc[:,['item_property_list','item_property_list','item_price_level','item_sales_level',
#                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
#                     'predict_category_property','shop_review_num_level',
#                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
#                     'shop_score_description']]

dataset2=new_feature_data.drop(['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_timestamp','context_page_id','shop_id'],axis=1,inplace=False)

label=data.loc[:,'is_trade']




###############lightGBM
#
#new_feature_data=new_feature_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))

#X=data.drop(['instance_id','context_id','is_trade'],axis=1,inplace=False)
X=new_feature_data

y=data[['is_trade']]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



import json
import lightgbm as lgb

from sklearn.metrics import roc_curve, auc, roc_auc_score


print("load data")
#df_train=pd.read_csv(path+"regression.train",header=None,sep='\t')
#df_test=pd.read_csv(path+"regression.train",header=None,sep='\t')
#y_train = df_train[0].values
#y_test = df_test[0].values
#X_train = df_train.drop(0, axis=1).values
#X_test = df_test.drop(0, axis=1).values
                     
df_train=X_train
#df_test=pd.read_csv(path+"regression.train",header=None,sep='\t')
y_train =y_train.iloc[:,0].values
y_test =y_test.iloc[:,0].values
X_train =np.array(X_train)
X_test = np.array(X_test)
                     
                     
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict

#
#
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'logloss', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}





#string2=['item_id','item_category_list','item_property_list','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_id','user_gender_id',
#         'user_age_level','user_occupation_id','user_star_level','context_timestamp','context_page_id','predict_category_property','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']

string2=new_feature_data.columns.values.tolist()
print(string2)
string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_timestamp','context_page_id','shop_id']









print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                feature_name=string2,
                categorical_feature=string3,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)



print('Save model...')
# save model to file
gbm.save_model('model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )


num_round = 300
lgb.cv(params, lgb_train, num_round, nfold=5,feature_name=string2,categorical_feature=string3,early_stopping_rounds=10)




fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')


from sklearn.metrics import log_loss
print(log_loss(y_test,y_pred))




import numpy as np
logloss=np.zeros((len(y_test),1))
p=y_pred
import math as math
for i in range(len(y_test)):
    logloss[i]=y_test[i]*(math.log(10,p[i]))+(1-y_test[i])*(math.log(10,1-p[i]))
    
logloss=-1/len(y_test)*(np.sum(logloss))
print('losloss is that',logloss)













##########进行独热编码
t1=dataset1.astype(str)
data_t1=pd.get_dummies(t1)
###########可以验证一下id类型的种类个数
###item:3695   item_brand_id:1101   item_city_id:99  user_id:13573   user_gender_id:4
####user_occupation_id:5  context_timestamp:5    shop_id:2015   item_category_list:13
t=pd.DataFrame(dataset2.loc[:,'item_category_list'])

t['instance_id_count']=1
t=t.groupby('item_category_list').agg('sum').reset_index()
#########



###########将数值型特征进行归一化
dataset2_ave=dataset2.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))


######将id类特征进行归一化

dataset1_ave=dataset1.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))


######输入GBDT进行特征融合
data_new=data.drop(['context_id','is_trade'],axis=1,inplace=False)
data_new=data_new.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))



X=data_new
y=label
#X=np.array(X)
#y=np.array(y)

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)

from sklearn.preprocessing import OneHotEncoder


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
#                                                            y_train,
#                                                            test_size=0.5)




#X_train_11=X_train[['item_id','item_category_list','item_brand_id','item_city_id','user_id',
#               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]
#               
#X_train_lr11=X_train[['item_id','item_category_list','item_brand_id','item_city_id','user_id',
#               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]    



grd_11 = GradientBoostingClassifier()
grd_enc_11 = OneHotEncoder()
grd_lm_11 = LogisticRegression()
grd_11.fit(X_train, y_train)

grd_enc_11.fit(grd_11.apply(X_train)[:, :, 0])

lr11=grd_enc_11.transform(grd_11.apply(X_train)[:, :, 0])



grd_lm_11.fit(lr11, y_train)


ltest11=grd_enc_11.transform(grd_11.apply(X_test)[:, :, 0])


y_pred_grd_lm = grd_lm_11.predict_proba(ltest11)[:, 1]
    
p=y_pred_grd_lm#############预测的概率

fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

xgb_lr_auc = roc_auc_score(y_test, y_pred_grd_lm)
print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')

logloss=np.zeros((len(y_test),1))
import math as math
for i in range(len(y_test)):
    logloss[i]=y_test[i]*(math.log(10,p[i]))+(1-y_test[i])*(math.log(10,1-p[i]))
    
logloss=-1/len(y_test)*(np.sum(logloss))
print('losloss is that',logloss)














#####对id类和非id类分别建树


X_train_1=X_train[['item_id']]
X_train_2=X_train[['item_category_list']]
X_train_3=X_train[['item_brand_id']]
X_train_4=X_train[['item_city_id']]
X_train_5=X_train[['user_id']]
X_train_6=X_train[['user_gender_id']]
X_train_7=X_train[['user_occupation_id']]
X_train_8=X_train[['context_timestamp']]
X_train_9=X_train[['shop_id']]
X_train_10=X_train[['item_property_list','item_property_list','item_price_level','item_sales_level',
                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
                     'context_page_id','predict_category_property','shop_review_num_level',
                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
                     'shop_score_description']]
                     

#X_train_lr1=X_train_lr[['item_id']]
#X_train_lr2=X_train_lr[['item_category_list']]
#X_train_lr3=X_train_lr[['item_brand_id']]
#X_train_lr4=X_train_lr[['item_city_id']]
#X_train_lr5=X_train_lr[['user_id']]
#X_train_lr6=X_train_lr[['user_gender_id']]
#X_train_lr7=X_train_lr[['user_occupation_id']]
#X_train_lr8=X_train_lr[['context_timestamp']]
#X_train_lr9=X_train_lr[['shop_id']]
#X_train_lr10=X_train_lr[['item_property_list','item_property_list','item_price_level','item_sales_level',
#                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
#                     'context_page_id','predict_category_property','shop_review_num_level',
#                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
#                     'shop_score_description']]

n_estimator =1

grd_1 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_1 = OneHotEncoder()
grd_lm_1 = LogisticRegression()
grd_1.fit(X_train_1, y_train)

grd_enc_1.fit(grd_1.apply(X_train_1)[:, :, 0])

lr1=grd_enc_1.transform(grd_1.apply(X_train_1)[:, :, 0])




grd_2 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_2 = OneHotEncoder()
grd_lm_2 = LogisticRegression()
grd_2.fit(X_train_2, y_train)

grd_enc_2.fit(grd_2.apply(X_train_2)[:, :, 0])

lr2=grd_enc_2.transform(grd_2.apply(X_train_2)[:, :, 0])


grd_3 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_3 = OneHotEncoder()
grd_lm_3 = LogisticRegression()
grd_3.fit(X_train_3, y_train)

grd_enc_3.fit(grd_3.apply(X_train_3)[:, :, 0])

lr3=grd_enc_3.transform(grd_3.apply(X_train_3)[:, :, 0])


grd_4 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_4 = OneHotEncoder()
grd_lm_4 = LogisticRegression()
grd_4.fit(X_train_4, y_train)

grd_enc_4.fit(grd_4.apply(X_train_4)[:, :, 0])

lr4=grd_enc_4.transform(grd_4.apply(X_train_4)[:, :, 0])



grd_5 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_5 = OneHotEncoder()
grd_lm_5 = LogisticRegression()
grd_5.fit(X_train_5, y_train)

grd_enc_5.fit(grd_5.apply(X_train_5)[:, :, 0])

lr5=grd_enc_5.transform(grd_5.apply(X_train_5)[:, :, 0])


grd_6 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_6 = OneHotEncoder()
grd_lm_6 = LogisticRegression()
grd_6.fit(X_train_6, y_train)

grd_enc_6.fit(grd_6.apply(X_train_6)[:, :, 0])

lr6=grd_enc_6.transform(grd_6.apply(X_train_6)[:, :, 0])



grd_7 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_7 = OneHotEncoder()
grd_lm_7 = LogisticRegression()
grd_7.fit(X_train_7, y_train)

grd_enc_7.fit(grd_7.apply(X_train_7)[:, :, 0])

lr7=grd_enc_7.transform(grd_7.apply(X_train_7)[:, :, 0])



grd_8 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_8 = OneHotEncoder()
grd_lm_8 = LogisticRegression()
grd_8.fit(X_train_8, y_train)

grd_enc_8.fit(grd_8.apply(X_train_8)[:, :, 0])

lr8=grd_enc_8.transform(grd_8.apply(X_train_8)[:, :, 0])


grd_9 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_9 = OneHotEncoder()
grd_lm_9 = LogisticRegression()
grd_9.fit(X_train_9, y_train)

grd_enc_9.fit(grd_9.apply(X_train_9)[:, :, 0])

lr9=grd_enc_9.transform(grd_9.apply(X_train_9)[:, :, 0])



grd_10 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_10 = OneHotEncoder()
grd_lm_10 = LogisticRegression()
grd_10.fit(X_train_10, y_train)

grd_enc_10.fit(grd_10.apply(X_train_10)[:, :, 0])

lr10=grd_enc_10.transform(grd_10.apply(X_train_10)[:, :, 0])








lr1=lr1.toarray()
lr2=lr2.toarray()
lr3=lr3.toarray()
lr4=lr4.toarray()
lr5=lr5.toarray()
lr6=lr6.toarray()
lr7=lr7.toarray()
lr8=lr8.toarray()
lr9=lr9.toarray()
lr10=lr10.toarray()



result=np.concatenate([lr1,lr2,lr3,lr4,lr5,lr6,lr7,lr8,lr9,lr10],axis=1)

y_train=np.array(y_train)

#######result为总的GBDT提取出来的特征向量
grd_lm=LogisticRegression()


grd_lm.fit(result, y_train)




##############测试集做同样处理
X_test_1=X_test[['item_id']]
X_test_2=X_test[['item_category_list']]
X_test_3=X_test[['item_brand_id']]
X_test_4=X_test[['item_city_id']]
X_test_5=X_test[['user_id']]
X_test_6=X_test[['user_gender_id']]
X_test_7=X_test[['user_occupation_id']]
X_test_8=X_test[['context_timestamp']]
X_test_9=X_test[['shop_id']]
X_test_10=X_test[['item_property_list','item_property_list','item_price_level','item_sales_level',
                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
                     'context_page_id','predict_category_property','shop_review_num_level',
                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
                     'shop_score_description']]


#
#grd_enc_9.fit(grd_9.apply(X_train_9)[:, :, 0])
#
#lr9=grd_enc_9.transform(grd_9.apply(X_train_lr9)[:, :, 0])


ltest1=grd_enc_1.transform(grd_1.apply(X_test_1)[:, :, 0])
ltest2=grd_enc_2.transform(grd_2.apply(X_test_2)[:, :, 0])
ltest3=grd_enc_3.transform(grd_3.apply(X_test_3)[:, :, 0])
ltest4=grd_enc_4.transform(grd_4.apply(X_test_4)[:, :, 0])
ltest5=grd_enc_5.transform(grd_5.apply(X_test_5)[:, :, 0])
ltest6=grd_enc_6.transform(grd_6.apply(X_test_6)[:, :, 0])
ltest7=grd_enc_7.transform(grd_7.apply(X_test_7)[:, :, 0])
ltest8=grd_enc_8.transform(grd_8.apply(X_test_8)[:, :, 0])
ltest9=grd_enc_9.transform(grd_9.apply(X_test_9)[:, :, 0])
ltest10=grd_enc_10.transform(grd_10.apply(X_test_10)[:, :, 0])

ltest1=ltest1.toarray()
ltest2=ltest2.toarray()
ltest3=ltest3.toarray()
ltest4=ltest4.toarray()
ltest5=ltest5.toarray()
ltest6=ltest6.toarray()
ltest7=ltest7.toarray()
ltest8=ltest8.toarray()
ltest9=ltest9.toarray()
ltest10=ltest10.toarray()


new_test=np.concatenate([ltest1,ltest2,ltest3,ltest4,ltest5,ltest6,ltest7,ltest8,ltest9,ltest10],axis=1)


y_test=np.array(y_test)


y_pred_grd_lm = grd_lm.predict_proba(new_test)[:, 1]
    
p=y_pred_grd_lm#############预测的概率

fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)



xgb_lr_auc = roc_auc_score(y_test, y_pred_grd_lm)
print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc)



import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')

logloss=np.zeros((len(y_test),1))

import math as math
for i in range(len(y_test)):
    logloss[i]=y_test[i]*(math.log(10,p[i]))+(1-y_test[i])*(math.log(10,1-p[i]))
    
logloss=-1/len(y_test)*(np.sum(logloss))
print('losloss is that',logloss)




#
#
#x1=pd.DataFrame(dataset1.loc[:,['user_id', 'user_gender_id']])
#x1['count']=1
#x1=x1.groupby(['user_id','user_gender_id']).agg('sum').reset_index()
#
#
#t1=dataset1.loc[:,['item_id']]
#t1=t1.apply(lambda x:x.astype(str))
#t11=pd.get_dummies(t1)
#    
#
##t1=np.array([[1],[2],[3],[5]])
##
##from sklearn import preprocessing
##enc=preprocessing.OneHotEncoder()
##enc.fit(t1)
##aa=enc.transform(t1)
##print(aa)
#
#t2=t1.loc[:,'item_id'].apply(lambda x :x<0)
#
#from numpy import argmax
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#
#d1=np.array(data[['item_id']])
#print(d1)
#
#label_encoder=LabelEncoder()
#integer_encoded=label_encoder.fit_transform(d1)
#
#
######广告商品的特征
#a13=t1
#t=a13
#t['instance_id_count']=1
#t=t.groupby('item_id').agg('sum').reset_index()
