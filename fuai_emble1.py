# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:05:16 2018

@author: 1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:35:14 2018

@author: 1
"""





import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")
from pandas import DataFrame as DF

data_train=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_train.txt',delimiter=' ',header=0)
#data_train=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)

data_train.shape
#data_train=data_train.drop('is_trade',axis=1)
#data_train.shape
data_test=pd.read_csv('D:/mother/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_a_20180425.txt',delimiter=' ',header=0)
#data_test=pd.read_csv('D:/mother/round1_ijcai_18_test_b_20180418.txt',delimiter=' ',header=0)

data_test.shape

#########删除训练数据的重复项
data_train.drop_duplicates(inplace=True)
data_train.dropna(inplace=True)
data_train.shape

data_test['is_trade']=10


#########统计含有空值的行数
row_null=data_test.isnull().sum(axis=1)
col_null=data_test.isnull().sum(axis=0)


def item_pro_split(x,num):
    if(len(x)>num):
        return x[num]
    else: 
        return x[0]

def data_origin(data):
    
####令商品的category——list为a12
    a1=data.item_category_list.map(lambda x:x.split(';'))
    a11=a1.apply(lambda x:x[0])
    a12=a1.apply(lambda x:x[1])
    a12.value_counts()
    data.item_category_list=a12
            
       ######广告商品属性列表数值化
    b1=data.item_property_list.apply(lambda x:x.split(';'))
    data['len of item_property']=b1.apply(lambda x:len(x))
    data['item_property_list_m1']=b1.apply(lambda x:item_pro_split(x,0))
    data['item_property_list_m2']=b1.apply(lambda x:item_pro_split(x,1))
    data['item_property_list_m3']=b1.apply(lambda x:item_pro_split(x,2))
    del data['item_property_list']
    
        #########时间戳时间格式化
    c2=data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    c3=c2.astype(str).apply(lambda x:x.split(' '))
    data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
    data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))
    #del data['context_timestamp']
    del data['context_id']
    
    ####查询词预测类目属性表数值化
    c1=data['predict_category_property'].apply(lambda x:x.split(';'))
    c11=c1.apply(lambda x:x[0].split(':')[0])
    data['len_of_predict_category']=c1.apply(lambda x:len(x))
    data['predict_category_property_m1']=c1.apply(lambda x:x[0].split(':')[0])
    del data['predict_category_property']
    del data['instance_id']
    data.item_category_list=data.item_category_list.apply(lambda x:int(x))
    data.item_property_list_m1=data.item_property_list_m1.apply(lambda x:int(x))
    data.item_property_list_m2=data.item_property_list_m2.apply(lambda x:int(x))
    data.item_property_list_m3=data.item_property_list_m3.apply(lambda x:int(x))
    data.predict_category_property_m1=data.predict_category_property_m1.apply(lambda x:int(x))
    
    return data

def get_division_feature(data,feature_name):
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns)-1):
        for j in range(i+1,len(data[feature_name].columns)):
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '*' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '+' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '-' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i]]/data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i]]*data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i]]+data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i]]-data[data[feature_name].columns[j]])
            
    
    temp_data = DF(pd.concat(new_feature,axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data,temp_data],axis=1).reset_index(drop=True)
    
    print(data.shape)
    
    return data.reset_index(drop=True)

def get_square_feature(data,feature_name):
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns)):
        new_feature_name.append(data[feature_name].columns[i] + '**2')
        new_feature_name.append(data[feature_name].columns[i] + '**1/2')
        new_feature.append(data[data[feature_name].columns[i]]**2)
        new_feature.append(data[data[feature_name].columns[i]]**(1/2))
        
    temp_data = DF(pd.concat(new_feature,axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data,temp_data],axis=1).reset_index(drop=True)
    
    print(data.shape)
    
    return data.reset_index(drop=True)


###########合并数据
data_train_test=pd.concat([data_train,data_test],ignore_index=True)
data1=data_origin(data_train_test)
del data_train
del data_train_test

##########生成商品的数值型特征的交叉特征
string_num=['item_price_level','item_sales_level','item_collected_level','item_pv_level']

num_data=data1[string_num]
num_square_data=get_square_feature(num_data,string_num)
for ss in string_num:
    del num_square_data[ss]
num_div_data=get_division_feature(num_data,string_num)
for ss in string_num:
    del num_div_data[ss]


########产生店铺的交叉特征
string_num1=['shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
num_data1=data1[string_num1]
num_square_data1=get_square_feature(num_data1,string_num1)
for ss in string_num1:
    del num_square_data1[ss]
num_div_data1=get_division_feature(num_data1,string_num1)
for ss in string_num1:
    del num_div_data1[ss]

##合并
data1=pd.concat([data1,num_square_data,num_div_data,num_square_data1,num_div_data1],axis=1)


del num_data
del num_square_data
del num_div_data
del num_data1
del num_square_data1
del num_div_data1


#########对一些类别变量进行one-hot
string_user=['user_gender_id','user_age_level','user_occupation_id','user_star_level']
for ss in string_user:
    temp=pd.get_dummies(data1[ss],prefix=ss)
    data1=pd.concat([data1,temp],axis=1)


del temp
del data1['user_gender_id']


#df_train=data1[data1.is_trade !=10]
#
###########查看每一天的交易情况
#import seaborn as sns
#import matplotlib.pyplot as plt
#plt.figure(figsize=(8,6),dpi=120)
#sns.countplot(y='day',hue='is_trade',data=df_train)
#
#
#del df_train
#data_y_train=data1[['day','is_trade']]

data=data1.copy()
del data1


######店铺的浏览次数
u1=data[['shop_id']]
u1['shop_id_see']=1
u1=u1.groupby('shop_id').agg('sum').reset_index()
shop_feature=u1

####user  and item
u1=data[['user_id','item_id']]
u1['user_item']=1
u1=u1.groupby(['user_id','item_id']).agg('sum').reset_index()
user_item=u1

#####iten brand  and shop
u1=data[['item_brand_id','shop_id']]
u1['item_brand_shop']=1
u1=u1.groupby(['item_brand_id','shop_id']).agg('sum').reset_index()
item_brand_shop=u1

#######item_property_list_m1是商品属性浏览次数
u1=data[['item_property_list_m1']]
u1['item_property_list_m1_see']=1
u1=u1.groupby(['item_property_list_m1']).agg('sum').reset_index()
item_property_list_m1_feature=u1


#####shop和day：在一天内的商店的编号的累加浏览次数
u1=data[['shop_id','day']]
u1['shop_day_times']=1
shop_day_sum=u1.groupby(['shop_id','day']).agg('sum').reset_index()


########uese and itemm在倚天的累加浏览次数
u1=data[['user_id','item_id','day']]
u1['user_item_day_times']=1
user_item_day_sum=u1.groupby(['user_id','item_id','day']).agg('sum').reset_index()




#######用户购买商品编号之间的时间差
u1=data[['user_id','item_id','context_timestamp']].sort_values(by=['user_id','item_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift'] = u1['user_id'].shift(1)
u1['item_shift'] = u1['item_id'].shift(1)
u1['time_delta_user_item'] = u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_shift !=u1.item_id),'time_delta_user_item']=-1
u1=u1.drop(['user_shift','item_shift'],axis=1)
time_delta_user_item=u1



#####用户操作距离上一条的时间差

u1=data[['user_id','context_timestamp']].sort_values(by=['user_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)

u1['user_shift']=u1['user_id'].shift(-1)
u1['before_time_delta_user']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[u1.user_shift!=u1.user_id,'before_time_delta_user']=1
u1=u1.drop('user_shift',axis=1)
before_time_delta_user=u1

######用户浏览category时间差
u1=data[['user_id','item_category_list','context_timestamp']].sort_values(by=['user_id','item_category_list','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['category_shift']=u1['item_category_list'].shift(-1)
u1['before_time_delta_user_cate']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.category_shift !=u1.item_category_list),'before_time_delta_user_cate']=1
u1=u1.drop(['user_shift','category_shift'],axis=1)
before_time_delta_user_cate=u1


###########---------------------商铺和上下文交叉----------------------
u1=data[['shop_id','hour']]
u1['shop_hour']=1
u1=u1.groupby(['shop_id','hour']).agg('sum').reset_index()
shop_hour=u1

##############用户浏览cate and pro_m1的时间差
u1=data[['user_id','item_category_list','item_property_list_m1','context_timestamp']].sort_values(by=['user_id','item_category_list','item_property_list_m1','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['item_category_list_shift']=u1['item_category_list'].shift(-1)
u1['item_propert_list_m1_shift']=u1['item_property_list_m1'].shift(-1)
u1['before_time_user_cate_pro_m1']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_category_list_shift !=u1.item_category_list)|(u1.item_propert_list_m1_shift !=u1.item_property_list_m1),'before_time_user_cate_pro_m1']=1
u1=u1.drop(['user_shift','item_category_list_shift','item_propert_list_m1_shift'],axis=1)
before_time_user_cate_pro_m1=u1


new_train_data=data.copy()
del data
new_train_data=pd.merge(new_train_data,shop_feature,on=['shop_id'],how='left')
del shop_feature
new_train_data=pd.merge(new_train_data,user_item,on=['user_id','item_id'],how='left')
del user_item
new_train_data=pd.merge(new_train_data,item_brand_shop,on=['item_brand_id','shop_id'],how='left')
del item_brand_shop
new_train_data=pd.merge(new_train_data,item_property_list_m1_feature,on=['item_property_list_m1'],how='left')
del item_property_list_m1_feature
new_train_data=pd.merge(new_train_data,shop_day_sum,on=['shop_id','day'],how='left')
del shop_day_sum
new_train_data=pd.merge(new_train_data,user_item_day_sum,on=['user_id','item_id','day'],how='left')
del user_item_day_sum


new_train_data=pd.merge(new_train_data,time_delta_user_item,on=['user_id','item_id','context_timestamp'],how='left')
del time_delta_user_item
new_train_data=pd.merge(new_train_data,before_time_delta_user,on=['user_id','context_timestamp'],how='left')
del before_time_delta_user
new_train_data=pd.merge(new_train_data,before_time_delta_user_cate,on=['user_id','item_category_list','context_timestamp'],how='left')
del before_time_delta_user_cate
new_train_data=pd.merge(new_train_data,shop_hour,on=['shop_id','hour'],how='left')
del shop_hour
new_train_data=pd.merge(new_train_data,before_time_user_cate_pro_m1,on=['user_id','item_category_list','item_property_list_m1','context_timestamp'],how='left')
del before_time_user_cate_pro_m1


del new_train_data['context_timestamp']

string2=new_train_data.columns.values.tolist()
print(string2)


yuce_data=new_train_data[new_train_data.is_trade==10]
tr_data=new_train_data[new_train_data.is_trade !=10]

string_data=tr_data.columns.values.tolist()
string_yuce=yuce_data.columns.values.tolist()


del new_train_data
del data_test

##########划分数据集
X_train=tr_data[(tr_data.day<7)|(tr_data.day==31)].drop(['day','is_trade'],axis=1,inplace=False)
X_test=tr_data[(tr_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)

y_train=tr_data[(tr_data.day<7)|(tr_data.day==31)][['is_trade']]
y_test=tr_data[(tr_data.day==7)][['is_trade']]

#
#
##############---------------lightGBM转换数据格式--------------------
import json
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import log_loss
#
#
print("load data")

#y_train =y_train.iloc[:,0].values
#y_test =y_test.iloc[:,0].values
#
#start_time=time.time()
#lgb_train=lgb.Dataset(X_train,y_train,free_raw_data=False)
#lgb_eval=lgb.Dataset(X_test,y_test,reference=lgb_train,free_raw_data=False)
#params={
#    'boosting_type': 'gbdt',
#    'objective': 'binary',
#    #'metric': 'binary_logloss',
#    'metric': 'auc',
#    'num_leaves': 31,
#    'learning_rate': 0.05,
#    'feature_fraction': 0.7,
#    'bagging_fraction': 1,
#    'bagging_freq': 10,
#    'verbose': 0, 
#    'min_data_in_leaf':10,
#    'feature_fraction_seed':seed,
#    'bagging_seed':seed,
#    'metric_freq':1 
#}

#string2=train.columns.values.tolist()
#print(string2)              
##string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id']
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_age_level','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']
###
#
#
lgb0 = lgb.LGBMClassifier(
        objective='binary',
#        metric='binary_logloss',

        num_leaves=255,
#        max_depth=6,
        learning_rate=0.025,
        seed=2018,
        reg_alpha=3,
#        reg_lambda=6,
        colsample_bytree=0.8,
#        min_child_weight=6,
        subsample=0.9,
        n_estimators=20000)


lgb_model = lgb0.fit(X_train, y_train, eval_set=[(X_test,y_test)],eval_metric='binary_logloss',early_stopping_rounds=100)
best_iter = lgb_model.best_iteration
#
############保存模型
from sklearn.externals import joblib
#joblib.dump(lgb_model,'fusai_7day_201852179808.pkl')
#
##lggb=joblib.load('fusai201851.pkl')
###############offline验证
off_line_test=tr_data[(tr_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)
off_line_y_test=tr_data[(tr_data.day==7)].is_trade
pred2_off= lgb_model.predict_proba(off_line_test)[:, 1]
print(log_loss(off_line_y_test,pred2_off))####0.1798083

pd.DataFrame(pred2_off).to_csv('ensamble/pre_off_light_179769.txt',index=False)


############求重要度
lgb.plot_importance(lgb_model)
predictors = [i for i in X_train.columns]
feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
print(feat_imp)
print(feat_imp.shape)

###########预测7号下半天
lggb=joblib.load('fusai_7day_201852179808.pkl')
best_iter = lggb.best_iteration




#######将7号上半天的添加进去

online_test=yuce_data[(yuce_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)
online_y_test=yuce_data[(yuce_data.day==7)][['is_trade']]

data_test=pd.read_csv('D:/mother/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_a_20180425.txt',delimiter=' ',header=0)


pred2= lgb_model.predict_proba(online_test)[:, 1]

bao_cun=data_test[['instance_id']]
bao_cun['predicted_score']=pred2
bao_cun.to_csv('round2_result/b201853_179769.txt',index=False,sep=' ')



##print("load data")
#X_train=tr_data[(tr_data.day<=7)|(tr_data.day==31)].drop(['day','is_trade'],axis=1,inplace=False)
#y_train=tr_data[(tr_data.day<=7)|(tr_data.day==31)][['is_trade]]
#y_train =y_train.iloc[:,0].values
##
##
##
#lgb0 = lgb.LGBMClassifier(
#        objective='binary',
#        num_leaves=127,
##        max_depth=7,
#        learning_rate=0.025,
#        seed=2018,
#        reg_alpha=5,
##        reg_lambda=6,
#        colsample_bytree=0.75,
##         min_child_samples=8,
#        subsample=0.9,
#        n_estimators=best_iter)
#
#lgb_model1 = lgb0.fit(X_train, y_train)

#pred2= lgb_model.predict_proba(online_test)[:, 1]
##
##
#bao_cun=data_test[['instance_id']]
#bao_cun['predicted_score']=pred2
#bao_cun.to_csv('b201852_LI798083.txt',index=False,sep=' ')



###########xgboost预测
import xgboost as xgb

dtrain1=xgb.DMatrix(X_train,y_train.is_trade)
dval=xgb.DMatrix(off_line_test, off_line_y_test)
dtest=xgb.DMatrix(online_test)
dval_test=xgb.DMatrix(off_line_test)
params={
        'booster':'gbtree',
        'objective':'binary:logistic',
#        'gamma':0.1
        'eta':0.05,
        'max_depth':8,
        'lambda':10,
        'subsample':0.9,
        'min_child_weight':5,
        'colsample_bytree':0.75,
        'silent':0,
        'eval_metric':'logloss',
        'nthread':12
        }


import time
start=time.clock()

watchlist=[(dtrain1,'train'),(dval,'val')]
mmodel=xgb.train(params,dtrain1,num_boost_round=20000,evals=watchlist,early_stopping_rounds=200)
sss=mmodel.best_iteration

elapsed=(time.clock()-start)
print("time used;",elapsed)

######线下
pres_offline=mmodel.predict(dval_test,ntree_limit=mmodel.best_iteration)
print(log_loss(y_test,pres_offline)) 

pre_online=mmodel.predict(dtest,ntree_limit=mmodel.best_iteration)

bao_cun=data_test[['instance_id']]
bao_cun['predicted_score']=pre_online
bao_cun.to_csv('xgb0180716.txt',index=False,sep=' ')

BAO_CHUN=pd.read_csv('xgb0180716.txt',delimiter=' ',header=0)
BAO_CHUN2=pd.read_csv('round2_result/b201851_18315.txt',delimiter=' ',header=0)

feature_score = mmodel.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))

with open('xgb0180716.csv','w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)














