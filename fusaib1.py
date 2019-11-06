# -*- coding: utf-8 -*-
"""
Created on Fri May 11 23:27:46 2018

@author: 1
"""



import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")
from pandas import DataFrame as DFs
import time
import random
import scipy.special as special
import math
from math import log

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        #更新策略
        for i in range(iter_num):
#            print(i)
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

#hyper = HyperParam(1, 1)
##这里的 I 和 C 代表总点击数和购买数，格式为pd.Series,如果要使用的话可以直接跳到倒数第二排的代码，把 I 和 C 改成某个商品历史的总点击数和被购买数的数据就可以了
#I, C = hyper.sample_from_beta(10, 1000, 10000, 1000)
#
#hyper.update_from_data_by_FPI(I, C, 1000, 0.00000001)
#print (hyper.beta,hyper.alpha)



data_train=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_train.txt',delimiter=' ',header=0)
#data_train=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_ijcai_18_test_b_20180510.txt',delimiter=' ',header=0)
#data_train=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)

#####时间戳格式化
c2=data_train.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
c3=c2.astype(str).apply(lambda x:x.split(' '))
data_train['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))


#data_day6=data_train[data_train.day==6]
data_train=data_train[(data_train.day==5)|(data_train.day==6)|(data_train.day==7)]

del c2
del c3
del data_train['day']



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

data_test=pd.read_csv('D:/mother/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_b_20180510.txt',delimiter=' ',header=0)
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
    
#        #########时间戳时间格式化
    c2=data.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
    c3=c2.astype(str).apply(lambda x:x.split(' '))
    data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
    data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))

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
########分割day6数据
#data_day6=data1[data1.day==6]
data_day5=data1[(data1.day==5)|(data1.day==6)]

data1=data1[data1.day==7]
###########计算第5天的点击率merge到第7天
string_id_cvr=['hour','item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','user_age_level','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']

for ss in string_id_cvr:
    hyper = HyperParam(1, 1)
    tmp=data_day5.groupby([ss])['is_trade'].agg({'count','sum'}).reset_index()
    I=tmp['count']
    C=tmp['sum']
    hyper.update_from_data_by_FPI(I, C, 1000, 0.00001)
#    print (hyper.beta,hyper.alpha)
    tmp[ss+'_cvr']=(C+hyper.alpha)/(I+hyper.beta+hyper.alpha)
    tmp=tmp.drop(['count','sum'],axis=1)
    data1=pd.merge(data1,tmp,on=ss,how='left')
    

data1.fillna(0,inplace=True)



#########对一些类别变量进行one-hot
string_user=['user_gender_id','user_occupation_id']
for ss in string_user:
    temp=pd.get_dummies(data1[ss],prefix=ss)
    data1=pd.concat([data1,temp],axis=1)


del temp
#del data1['user_gender_id']
#del data1['user_occupation_id']
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
#u1=data[['shop_id','day']]
#u1['shop_day_times']=1
#shop_day_sum=u1.groupby(['shop_id','day']).agg('sum').reset_index()
#
#
#########uese and itemm在倚天的累加浏览次数
#u1=data[['user_id','item_id','day']]
#u1['user_item_day_times']=1
#user_item_day_sum=u1.groupby(['user_id','item_id','day']).agg('sum').reset_index()




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
#del data
new_train_data=pd.merge(new_train_data,shop_feature,on=['shop_id'],how='left')
del shop_feature
new_train_data=pd.merge(new_train_data,user_item,on=['user_id','item_id'],how='left')
del user_item
new_train_data=pd.merge(new_train_data,item_brand_shop,on=['item_brand_id','shop_id'],how='left')
del item_brand_shop
new_train_data=pd.merge(new_train_data,item_property_list_m1_feature,on=['item_property_list_m1'],how='left')
del item_property_list_m1_feature
#new_train_data=pd.merge(new_train_data,shop_day_sum,on=['shop_id','day'],how='left')
#del shop_day_sum
#new_train_data=pd.merge(new_train_data,user_item_day_sum,on=['user_id','item_id','day'],how='left')
#del user_item_day_sum


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




########新加特征
###每个用户买了几种item_id
for ss in ['item_id','item_category_list','item_property_list_m1','item_brand_id']:
    u1=data.groupby(['user_id'])[ss].count().reset_index()
    u1.columns=['user_id','sum_'+ss]
    new_train_data=pd.merge(new_train_data,u1,on='user_id',how='left')


####每个商品倍多少用户购买
for ss in ['shop_id','item_id','item_category_list','item_property_list_m1','item_brand_id']:
    u1=data.groupby(ss)['user_id'].count().reset_index()
    u1.columns=[ss,'sum_'+ss+'user_id']
    new_train_data=pd.merge(new_train_data,u1,on=ss,how='left')


#for ss in ['shop_id','item_id','item_category_list','item_property_list_m1','item_brand_id']:
#    u1=data.groupby([ss,'day'])['user_id'].count().reset_index()
#    u1.columns=[ss,'day','sum_'+ss+'day'+'user_id']
#    new_train_data=pd.merge(new_train_data,u1,on=[ss,'day'],how='left')




string2=new_train_data.columns.values.tolist()
print(string2)


yuce_data=new_train_data[new_train_data.is_trade==10]
tr_data=new_train_data[new_train_data.is_trade !=10]

string_data=tr_data.columns.values.tolist()
string_yuce=yuce_data.columns.values.tolist()


del new_train_data
del data_test

##########划分数据集
#X_train=tr_data[(tr_data.day<7)|(tr_data.day==31)].drop(['day','is_trade'],axis=1,inplace=False)
#X_test=tr_data[(tr_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)
#
#y_train=tr_data[(tr_data.day<7)|(tr_data.day==31)][['is_trade']]
#y_test=tr_data[(tr_data.day==7)][['is_trade']]

#
#
################---------------lightGBM转换数据格式--------------------
import json
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
######划分训练与验证
#train_day7m,valxy=train_test_split(tr_data[tr_data.day==7],test_size=0.5,random_state=2018,stratify=tr_data[tr_data.day==7].is_trade)

#trainxy=pd.concat([tr_data[tr_data.day==6],train_day7m])
#del train_day7m

trainxy=tr_data[tr_data.day==7]
valxy=tr_data[tr_data.day==7]


#trainxy,valxy=train_test_split(tr_data,test_size=0.1,random_state=2018,stratify=tr_data.is_trade)
trainxy.loc[trainxy.hour==0,'hour']=24
valxy.loc[valxy.hour==0,'hour']=24


X_train=trainxy.drop(['hour','context_timestamp','day','is_trade'],axis=1,inplace=False)
y_train=trainxy.is_trade

X_test=valxy.drop(['hour','context_timestamp','day','is_trade'],axis=1,inplace=False)
y_test=valxy.is_trade

tr_string=X_train.columns.values.tolist()
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_age_level','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']
string_drop=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']

#for ss in string_drop:
#    del X_train[ss]
#    del X_test[ss]


#####数据切换
lgb_train=lgb.Dataset(X_train,y_train,free_raw_data=False)
lgb_eval=lgb.Dataset(X_test,y_test,free_raw_data=False)

params={
        'boosting_type':'gbdt',
        'objective':'binary',
        'metric':'binary_logloss',
        'num_leaves':6,
        'learning_rate':0.05,
        'feature_fraction':0.75,
        'bagging_fraction':0.9,
        'lambda_l1':5,
#        'min_child_weigth':10,
        'unbalance':True
        }

####10.170.61312

bst=lgb.cv(params,lgb_train,seed=2018,num_boost_round=10000,nfold=10,feature_name=tr_string,categorical_feature=string_drop,early_stopping_rounds=50,verbose_eval=True)
##bst=lgb.cv(params,lgb_train,seed=2018,num_boost_round=10000,nfold=5,early_stopping_rounds=50,verbose_eval=True)

num_boost_round=len(bst['binary_logloss-mean'])
print('best_num_boost:',num_boost_round,'   best_logloss:',bst['binary_logloss-mean'][num_boost_round-1])
######用eval验证
#bst=lgb.train(params,lgb_train,feature_name=tr_string,categorical_feature=string_drop,early_stopping_rounds=100)

bst1=lgb.train(params,lgb_train,num_boost_round,feature_name=tr_string,categorical_feature=string_drop)

#
#pre_off=bst1.predict(X_test)
#log_loss(y_test,pre_off)
#
#
#
######on_line 预测
#online_test=yuce_data.drop(['context_timestamp','day','is_trade'],axis=1,inplace=False)
##for ss in string3:
##        del online_test[ss]
##pre_on=lgb.predict(online_test,num_iteration=lgb.best_iteration)
#pre_on=bst1.predict(online_test)
#
#pre_on=pd.Series(pre_on).apply(lambda x:float("%.6f" % x))
#
#
#
#def jiao(x):
#    if x<0.01:
#        return math.sqrt(x)
#    else:
#        return x
#
#aa=pre_on.apply(lambda x:jiao(x)).apply(lambda x:"%.6f" % x)
#aa=aa.apply(lambda x:float(x))
#
#data_test=pd.read_csv('D:/mother/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_b_20180510.txt',delimiter=' ',header=0)
#
#bao_cun=data_test[['instance_id']]
#bao_cun['predicted_score']=pre_on
#bao_cun.to_csv('round2_result/lgb_017063221649.txt',index=False,sep=' ')
#


#estimators=lgb.train(params,lgb_train,num_boost_round)
#
####查看重要性
importance=bst1.feature_importance()
names=bst1.feature_name()
feat_imp = pd.Series(importance,names).sort_values(ascending=False)
print(feat_imp)
print(feat_imp.shape)


############预测7号下半天
#lggb=joblib.load('fusai_7day_201852179808.pkl')
#best_iter = lggb.best_iteration
#
#
#
#
########将7号上半天的添加进去
#
#online_test=yuce_data[(yuce_data.day==7)].drop(['day','is_trade'],axis=1,inplace=False)
#online_y_test=yuce_data[(yuce_data.day==7)][['is_trade']]
#
#data_test=pd.read_csv('D:/mother/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_a_20180425.txt',delimiter=' ',header=0)
#
#
#pred2= lgb_model.predict_proba(online_test)[:, 1]
#
#bao_cun=data_test[['instance_id']]
#bao_cun['predicted_score']=pred2
#bao_cun.to_csv('round2_result/b201853_179769.txt',index=False,sep=' ')



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


#
############xgboost预测
#import xgboost as xgb
#
#dtrain1=xgb.DMatrix(X_train,y_train.is_trade)
#dval=xgb.DMatrix(off_line_test, off_line_y_test)
#dtest=xgb.DMatrix(online_test)
#dval_test=xgb.DMatrix(off_line_test)
#params={
#        'booster':'gbtree',
#        'objective':'binary:logistic',
##        'gamma':0.1
#        'eta':0.05,
#        'max_depth':8,
#        'lambda':10,
#        'subsample':0.9,
#        'min_child_weight':5,
#        'colsample_bytree':0.75,
#        'silent':0,
#        'eval_metric':'logloss',
#        'nthread':12
#        }
#
#
#import time
#start=time.clock()
#
#watchlist=[(dtrain1,'train'),(dval,'val')]
#mmodel=xgb.train(params,dtrain1,num_boost_round=20000,evals=watchlist,early_stopping_rounds=200)
#sss=mmodel.best_iteration
#
#elapsed=(time.clock()-start)
#print("time used;",elapsed)
#
#######线下
#pres_offline=mmodel.predict(dval_test,ntree_limit=mmodel.best_iteration)
#print(log_loss(y_test,pres_offline)) 
#
#pre_online=mmodel.predict(dtest,ntree_limit=mmodel.best_iteration)
#
#bao_cun=data_test[['instance_id']]
#bao_cun['predicted_score']=pre_online
#bao_cun.to_csv('xgb0180716.txt',index=False,sep=' ')
#
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
#












