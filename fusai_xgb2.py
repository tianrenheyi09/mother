# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:59:39 2018

@author: 1
"""

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
import time

start=time.clock()

#reader=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_train.txt',delimiter=' ',header=0,iterator=True)
data_train=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_train.txt',delimiter=' ',header=0)
#data_train=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)


#try:
#    df=reader.get_chunk(1000000)
#except StopIteration:
#    print('Iteration id stopped')
#
#loop=True
#chunkSize=1000000
#chunks=[]
#while loop:
#    try:
#        chunk=reader.get_chunk(chunkSize)
#        chunks.append(chunk)
#    except StopIteration:
#        loop=False
#        print('Itera is stopeed')
#
#df=pd.concat(chunks,ignore_index=True)
#
#elapsed=(time.clock()-start)
#print("time used;",elapsed)
#del chunk
#
#df.shape
#data_train=data_train.drop('is_trade',axis=1)
data_train.shape

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
            new_feature.append(data[data[feature_name].columns[i]]/(0.5+data[data[feature_name].columns[j]]))
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
data1=pd.concat([data_train,data_test],ignore_index=True)
data1=data_origin(data1)
del data_train

data1[(data1.day==7)&(data1.is_trade !=10)][['is_trade']].to_csv('D:/mother/ensamble/day7_trade.txt',index=False)


#########查看合并后的类别情况
data1.item_price_level.value_counts()

data1.item_sales_level.value_counts()

data1.loc[data1.item_sales_level==-1,'item_sales_level']=0

data1.item_collected_level.value_counts()
data1.item_pv_level.value_counts()
data1.item_city_id.value_counts()

data1.user_gender_id.value_counts()##由-1
data1.user_occupation_id.value_counts()#由-1
data1.user_star_level.value_counts()##由-1

data1.loc[data1.user_star_level==-1,'user_star_level']=3006

data1.user_age_level.value_counts()
data1.loc[data1.user_age_level==-1,'user_age_level']=1003
#aa=data_train.user_age_level.replace(-1,1003)######替换为最多众数

data1.context_page_id.value_counts()
data1.shop_review_num_level.value_counts()
data1.shop_star_level.value_counts()
print('填充异常值完毕')

###########生成商品的数值型特征的交叉特征
#string_num=['item_price_level','item_sales_level','item_collected_level','item_pv_level']
#
#num_data=data1[string_num]
#num_square_data=get_square_feature(num_data,string_num)
#for ss in string_num:
#    del num_square_data[ss]
#num_div_data=get_division_feature(num_data,string_num)
#for ss in string_num:
#    del num_div_data[ss]
#
#
#########产生店铺的交叉特征
#string_num1=['shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
#num_data1=data1[string_num1]
#num_square_data1=get_square_feature(num_data1,string_num1)
#for ss in string_num1:
#    del num_square_data1[ss]
#num_div_data1=get_division_feature(num_data1,string_num1)
#for ss in string_num1:
#    del num_div_data1[ss]

##合并
#data1=pd.concat([data1,num_square_data,num_div_data,num_square_data1,num_div_data1],axis=1)
#
#
#del num_data
#del num_square_data
#del num_div_data
#del num_data1
#del num_square_data1
#del num_div_data1


#########对一些类别变量进行one-hot
string_user=['user_gender_id','user_occupation_id']
for ss in string_user:
    temp=pd.get_dummies(data1[ss],prefix=ss)
    data1=pd.concat([data1,temp],axis=1)


del temp
del data1['user_gender_id']
del data1['user_occupation_id']


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


#######--------------------------用户根据什么等级购买
#########user and price level
u1=data[['user_id','item_price_level']]
u1['user_pric_lev']=1
user_pric_lev=u1.groupby(['user_id','item_price_level']).agg('sum').reset_index()

######user adn sale level
u1=data[['user_id','item_sales_level']]
u1['user_sale_lev']=1
user_sale_lev=u1.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
#######user and item_collevt
u1=data[['user_id','item_collected_level']]
u1['user_coll_lev']=1
user_coll_lev=u1.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
#########商品便宜是不是购买人多
u1=data[['item_category_list']]
u1['item_cate_see']=1
item_cate_see=u1.groupby(['item_category_list']).agg('sum').reset_index()
#########item ccate  and price level
u1=data[['item_category_list','item_price_level']]
u1['item_cate_price']=1
item_cate_price=u1.groupby(['item_category_list','item_price_level']).agg('sum').reset_index()


#------------------
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
new_train_data=pd.merge(new_train_data,user_pric_lev,on=['user_id','item_price_level'],how='left')
del user_pric_lev
new_train_data=pd.merge(new_train_data,user_sale_lev,on=['user_id','item_sales_level'],how='left')
del user_sale_lev
new_train_data=pd.merge(new_train_data,user_coll_lev,on=['user_id','item_collected_level'],how='left')
del user_coll_lev
new_train_data=pd.merge(new_train_data,item_cate_see,on=['item_category_list'],how='left')
del item_cate_see

new_train_data=pd.merge(new_train_data,item_cate_price,on=['item_category_list','item_price_level'],how='left')
del item_cate_price


#######--------------------------------
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


##########划分数据集
#train=tr_data[(tr_data.day<24)].drop(['day','is_trade'],axis=1,inplace=False)
#test=tr_data[(tr_data.day==24)].drop(['day','is_trade'],axis=1,inplace=False)
#y_train=tr_data[(tr_data.day<24)].is_trade
#y_test=tr_data[(tr_data.day==24)].is_trade


#

X_train=tr_data[(tr_data.day<7)|(tr_data.day==31)].drop(['day','is_trade'],axis=1,inplace=False)
X_test=tr_data[(tr_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)

y_train=tr_data[(tr_data.day<7)|(tr_data.day==31)].is_trade
y_test=tr_data[(tr_data.day==7)].is_trade


train_string=X_train.columns.values.tolist()
for ss in train_string:
    if (X_train[ss].dtypes=='object'):
        del X_train[ss]
        del X_test[ss]
#
##############---------------lightGBM转换数据格式--------------------
import json
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import log_loss
#
#
print("load data")

###############offline验证
off_line_test=tr_data[(tr_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)
off_line_y_test=tr_data[(tr_data.day==7)].is_trade



#######将7号上半天的添加进去


lgb0 = lgb.LGBMClassifier(
        objective='binary',
#        metric='binary_logloss',

        num_leaves=255,
#        max_depth=8,
        learning_rate=0.025,
        seed=2018,
        reg_alpha=1,
#        reg_lambda=6,
        colsample_bytree=0.75,
#        min_child_weight=6,
        subsample=0.9,
        n_estimators=20000)


lgb_model = lgb0.fit(X_train, y_train, eval_set=[(X_test,y_test)],eval_metric='binary_logloss',early_stopping_rounds=100)
best_iter = lgb_model.best_iteration
#













#############xgboost预测
import xgboost as xgb

dtrain1=xgb.DMatrix(X_train,y_train)
dval=xgb.DMatrix(X_test,y_test)
#dtest=xgb.DMatrix(online_test)
#dval_test=xgb.DMatrix(off_line_test)
params={
        'booster':'gbtree',
        'objective':'binary:logistic',
#        'gamma':0.1
        'eta':0.05,
        'max_depth':15,
        'lambda':3,
        'subsample':0.9,
        'min_child_weight':5,
        'colsample_bytree':0.8,
        'silent':0,
        'eval_metric':'logloss',
        'nthread':12
        }


import time
start=time.clock()

watchlist=[(dtrain1,'train'),(dval,'val')]
mmodel=xgb.train(params,dtrain1,num_boost_round=20000,evals=watchlist,early_stopping_rounds=10)
sss=mmodel.best_iteration
##
elapsed=(time.clock()-start)
print("time used;",elapsed)

########线下
pres_offline=mmodel.predict(dval_test,ntree_limit=mmodel.best_iteration)
print(log_loss(y_test,pres_offline))
 
pd.DataFrame(pres_offline).to_csv('ensamble/pre_off_xgb_178808.txt',index=False)


#pre_online=mmodel.predict(dtest,ntree_limit=mmodel.best_iteration)
#
#data_test=pd.read_csv('D:/mother/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_a_20180425.txt',delimiter=' ',header=0)
#
#bao_cun=data_test[['instance_id']]
#bao_cun['predicted_score']=pre_online
#bao_cun.to_csv('round2_result/xgb0178805.txt',index=False,sep=' ')

#BAO_CHUN=pd.read_csv('xgb0180716.txt',delimiter=' ',header=0)
#BAO_CHUN2=pd.read_csv('round2_result/b201851_18315.txt',delimiter=' ',header=0)
#
#feature_score = mmodel.get_fscore()
#feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
#fs = []
#for (key,value) in feature_score:
#    fs.append("{0},{1}\n".format(key,value))
#
#with open('xgb0180716.csv','w') as f:
#        f.writelines("feature,score\n")
#        f.writelines(fs)
#













