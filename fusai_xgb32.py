# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:41:15 2018

@author: 1
"""




import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")
from pandas import DataFrame as DF
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


#reader=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_train.txt',delimiter=' ',header=0,iterator=True)
data_train=pd.read_csv('D:/mother/[update] round2_ijcai_18_train_20180425/round2_train.txt',delimiter=' ',header=0)
#data_train=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)

#####时间戳格式化
c2=data_train.context_timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
c3=c2.astype(str).apply(lambda x:x.split(' '))
data_train['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))


data_train=data_train[(data_train.day==6)|(data_train.day==7)]
del c2
del c3
del data_train['day']


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
data_day6=data1[data1.day==6]
data1=data1[data1.day==7]
###########计算第6天的点击率merge到第7天
string_id_cvr=['hour','item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','user_age_level','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']

for ss in string_id_cvr:
    hyper = HyperParam(1, 1)
    tmp=data_day6.groupby([ss])['is_trade'].agg({'count','sum'}).reset_index()
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



#df_train=data1[data1.is_trade !=10]
#
###########查看每一天的交易情况
#import seaborn as sns
#import matplotlib.pyplot as plt
#plt.figure(figsize=(8,6),dpi=120)
#sns.countplot(y='day',hue='is_trade',data=tr_data)
#
#
#del df_train
#data_y_train=data1[['day','is_trade']]

data=data1.copy()
del data1


#######--------------------------用户根据什么等级购买
#########user and price level
#u1=data[['user_id','item_price_level']]
#u1['user_pric_lev']=1
#user_pric_lev=u1.groupby(['user_id','item_price_level']).agg('sum').reset_index()
#
#######user adn sale level
#u1=data[['user_id','item_sales_level']]
#u1['user_sale_lev']=1
#user_sale_lev=u1.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
########user and item_collevt
#u1=data[['user_id','item_collected_level']]
#u1['user_coll_lev']=1
#user_coll_lev=u1.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
#########商品便宜是不是购买人多
#u1=data[['item_category_list']]
#u1['item_cate_see']=1
#item_cate_see=u1.groupby(['item_category_list']).agg('sum').reset_index()
#########item ccate  and price level
u1=data[['item_category_list','item_price_level']]
u1['item_cate_price']=1
item_cate_price=u1.groupby(['item_category_list','item_price_level']).agg('sum').reset_index()
######ite cate  and sles
u1=data[['item_category_list','item_sales_level']]
u1['item_cate_sales']=1
item_cate_sales=u1.groupby(['item_category_list','item_sales_level']).agg('sum').reset_index()

#######item and price
u1=data[['item_id','item_price_level']]
u1['item_price']=1
item_price=u1.groupby(['item_id','item_price_level']).agg('sum').reset_index()
#####item and sale 
u1=data[['item_id','item_sales_level']]
u1['item_sales']=1
item_sales=u1.groupby(['item_id','item_sales_level']).agg('sum').reset_index()
######item and collect
u1=data[['item_id','item_collected_level']]
u1['item_coll_lev']=1
item_coll_lev=u1.groupby(['item_id','item_collected_level']).agg('sum').reset_index()



#--------------------------------item_id特征----------------------------
all_item_data=data
string_dan_item=['item_id','item_category_list','item_property_list_m1','item_property_list_m2','item_property_list_m3',
             'item_brand_id']

for ss in string_dan_item:
    u1=data[[ss]]
    u1[ss+'see']=1
    u1=u1.groupby([ss]).agg('sum').reset_index()
    all_item_data=pd.merge(all_item_data,u1,on=ss,how='left')

all_item_data=pd.merge(all_item_data,item_cate_price,on=['item_category_list','item_price_level'],how='left')
del item_cate_price
all_item_data=pd.merge(all_item_data,item_cate_sales,on=['item_category_list','item_sales_level'],how='left')
del item_cate_sales
all_item_data=pd.merge(all_item_data,item_price,on=['item_id','item_price_level'],how='left')
del item_price
all_item_data=pd.merge(all_item_data,item_sales,on=['item_id','item_sales_level'],how='left')
del item_sales
all_item_data=pd.merge(all_item_data,item_coll_lev,on=['item_id','item_collected_level'],how='left')
del item_coll_lev

######item_id与categoty交互浏览次数
u1=data[['item_id','item_category_list']]
u1['item_item_category_see']=1
u1=u1.groupby(['item_id','item_category_list']).agg('sum').reset_index()
item_item_category_see=u1

######category and property m1交互次数
u1=data[['item_category_list','item_property_list_m1']]
u1['category_pro_m1_see']=1
u1=u1.groupby(['item_category_list','item_property_list_m1']).agg('sum').reset_index()
category_pro_m1_see=u1

######category and property m2交互次数
u1=data[['item_category_list','item_property_list_m2']]
u1['category_pro_m2_see']=1
u1=u1.groupby(['item_category_list','item_property_list_m2']).agg('sum').reset_index()
category_pro_m2_see=u1

######category and property m3交互次数
u1=data[['item_category_list','item_property_list_m3']]
u1['category_pro_m3_see']=1
u1=u1.groupby(['item_category_list','item_property_list_m3']).agg('sum').reset_index()
category_pro_m3_see=u1

######item_id与item_brand交互
u1=data[['item_id','item_brand_id']]
u1['item_item_brand_see']=1
u1=u1.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
item_item_brand_see=u1

######item_brand与category交互
u1=data[['item_brand_id','item_category_list']]
u1['brand_categoty_see']=1
u1=u1.groupby(['item_brand_id','item_category_list']).agg('sum').reset_index()
brand_categoty_see=u1

#####item_brand与property m1交互次数
u1=data[['item_brand_id','item_property_list_m1']]
u1['brand_pro_m1_see']=1
u1=u1.groupby(['item_brand_id','item_property_list_m1']).agg('sum').reset_index()
brand_pro_m1_see=u1

#####item_brand与property m2交互次数
u1=data[['item_brand_id','item_property_list_m2']]
u1['brand_pro_m2_see']=1
u1=u1.groupby(['item_brand_id','item_property_list_m2']).agg('sum').reset_index()
brand_pro_m2_see=u1

#####item_brand与property m3交互次数
u1=data[['item_brand_id','item_property_list_m3']]
u1['brand_pro_m3_see']=1
u1=u1.groupby(['item_brand_id','item_property_list_m3']).agg('sum').reset_index()
brand_pro_m3_see=u1


all_item_data=pd.merge(all_item_data,item_item_category_see,on=['item_id','item_category_list'],how='left')
del item_item_category_see
all_item_data=pd.merge(all_item_data,category_pro_m1_see,on=['item_category_list','item_property_list_m1'],how='left')
del category_pro_m1_see
all_item_data=pd.merge(all_item_data,category_pro_m2_see,on=['item_category_list','item_property_list_m2'],how='left')
del category_pro_m2_see
all_item_data=pd.merge(all_item_data,category_pro_m3_see,on=['item_category_list','item_property_list_m3'],how='left')
del category_pro_m3_see
all_item_data=pd.merge(all_item_data,item_item_brand_see,on=['item_id','item_brand_id'],how='left')
del item_item_brand_see
all_item_data=pd.merge(all_item_data,brand_categoty_see,on=['item_brand_id','item_category_list'],how='left')
del brand_categoty_see
all_item_data=pd.merge(all_item_data,brand_pro_m1_see,on=['item_brand_id','item_property_list_m1'],how='left')
del brand_pro_m1_see
all_item_data=pd.merge(all_item_data,brand_pro_m2_see,on=['item_brand_id','item_property_list_m2'],how='left')
del brand_pro_m2_see
all_item_data=pd.merge(all_item_data,brand_pro_m3_see,on=['item_brand_id','item_property_list_m3'],how='left')
del brand_pro_m3_see

all_item_data=all_item_data.fillna(0)
#all_item_data.to_csv('all_item_data.csv',index=None)
string_column=data.columns.values.tolist()
for ss in string_column:
    del all_item_data[ss]

##############------------------用户相关特征----------------
all_user_data=data
string_user=['user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']
for ss in string_user:
    u1=data[[ss]]
    u1[ss+'_see']=1
    u1=u1.groupby(ss).agg('sum').reset_index()
    all_user_data=pd.merge(all_user_data,u1,on=ss,how='left')
    
#####gender_age交叉
u1=data[['user_gender_id','user_age_level']]
u1['user_gender_age']=1
u1=u1.groupby(['user_gender_id','user_age_level']).agg('sum').reset_index()
user_gender_age=u1
####gender_ocu交叉
u1=data[['user_gender_id','user_occupation_id']]
u1['user_gender_ocu']=1
u1=u1.groupby(['user_gender_id','user_occupation_id']).agg('sum').reset_index()
user_gender_ocu=u1
####age与ocu交叉
u1=data[['user_age_level','user_occupation_id']]
u1['user_age_ocu']=1
u1=u1.groupby(['user_age_level','user_occupation_id']).agg('sum').reset_index()
user_age_ocu=u1


all_user_data=pd.merge(all_user_data,user_gender_age,on=['user_gender_id','user_age_level'],how='left')
del user_gender_age
all_user_data=pd.merge(all_user_data,user_gender_ocu,on=['user_gender_id','user_occupation_id'],how='left')
del user_gender_ocu
all_user_data=pd.merge(all_user_data,user_age_ocu,on=['user_age_level','user_occupation_id'],how='left')
del user_age_ocu
all_user_data=all_user_data.fillna(0)
#all_user_data.to_csv('all_user_data.csv',index=None)
string_column=data.columns.values.tolist()
for ss in string_column:
    del all_user_data[ss]


#########----------------上下文特征--------------
#string_page=['context_page_id','predict_category_property_m1']
#
#all_page_data=data
#for ss in string_page:
#    u1=data[[ss]]
#    u1[ss+'_see']=1
#    u1=u1.groupby(ss).agg('sum').reset_index()
#    all_page_data=pd.merge(all_page_data,u1,on=ss,how='left')
#
#######page and cate
#u1=data[['context_page_id','predict_category_property_m1']]
#u1['page_cate_m1']=1
#u1=u1.groupby(['context_page_id','predict_category_property_m1']).agg('sum').reset_index()
#page_cate_m1=u1
#
#
#all_page_data=pd.merge(all_page_data,page_cate_m1,on=['context_page_id','predict_category_property_m1'],how='left')
#del page_cate_m1
#
#all_page_data=all_page_data.fillna(0)
##all_page_data.to_csv('all_page_data.csv',index=None)
#string_column=data.columns.values.tolist()
#for ss in string_column:
#    del all_page_data[ss]

############--------------------店铺特征------------------
string_shop=['shop_id','shop_review_num_level','shop_star_level']
all_shop_data=data
for ss in string_shop:
    u1=data[[ss]]
    u1[ss+'_see']=1
    u1=u1.groupby(ss).agg('sum').reset_index()
    all_shop_data=pd.merge(all_shop_data,u1,on=ss,how='left')


####新的店铺特征
u1=data[['shop_id','shop_review_num_level']]
u1['shop_rev']=1
shop_rev=u1.groupby(['shop_id','shop_review_num_level']).agg('sum').reset_index()
#####shop and str
u1=data[['shop_id','shop_star_level']]
u1['shop_star']=1
shop_star=u1.groupby(['shop_id','shop_star_level']).agg('sum').reset_index()

all_shop_data=pd.merge(all_shop_data,shop_rev,on=['shop_id','shop_review_num_level'],how='left')
del shop_rev
all_shop_data=pd.merge(all_shop_data,shop_star,on=['shop_id','shop_star_level'],how='left')
del shop_star

all_shop_data=all_shop_data.fillna(0)
#all_shop_data.to_csv('all_shop_data.csv',index=None)
string_column=data.columns.values.tolist()
for ss in string_column:
    del all_shop_data[ss]

##############--------------------商品和用户的交叉特征-----------
####user  and item
u1=data[['user_id','item_id']]
u1['user_item']=1
u1=u1.groupby(['user_id','item_id']).agg('sum').reset_index()
user_item=u1
#####user  and item_category
u1=data[['user_id','item_category_list']]
u1['user_item_cate']=1
u1=u1.groupby(['user_id','item_category_list']).agg('sum').reset_index()
user_item_cate=u1

######user and item_pro_m1
u1=data[['user_id','item_property_list_m1']]
u1['user_item_pro_m1']=1
u1=u1.groupby(['user_id','item_property_list_m1']).agg('sum').reset_index()
user_item_pro_m1=u1

######user and item_pro_m2
u1=data[['user_id','item_property_list_m2']]
u1['user_item_pro_m2']=1
u1=u1.groupby(['user_id','item_property_list_m2']).agg('sum').reset_index()
user_item_pro_m2=u1
######user and item_pro_m1
u1=data[['user_id','item_property_list_m3']]
u1['user_item_pro_m3']=1
u1=u1.groupby(['user_id','item_property_list_m3']).agg('sum').reset_index()
user_item_pro_m3=u1
#####user and item brand
u1=data[['user_id','item_brand_id']]
u1['user_item_brand']=1
u1=u1.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
user_item_brand=u1


#############用户和数值型的特征交叉
u1=data[['user_id','item_price_level']]
u1['user_item_price']=1
u1=u1.groupby(['user_id','item_price_level']).agg('sum').reset_index()
user_item_price=u1
#############user and sales
u1=data[['user_id','item_sales_level']]
u1['user_item_sales']=1
u1=u1.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
user_item_sales=u1
############user and collected
u1=data[['user_id','item_collected_level']]
u1['user_item_collected']=1
u1=u1.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
user_item_collected=u1
##########user and pv
u1=data[['user_id','item_pv_level']]
u1['user_item_pv']=1
u1=u1.groupby(['user_id','item_pv_level']).agg('sum').reset_index()
user_item_pv=u1
##########user and city
u1=data[['user_id','item_city_id']]
u1['user_item_city']=1
u1=u1.groupby(['user_id','item_city_id']).agg('sum').reset_index()
user_item_city=u1




#################-----------用户和上下文的特征-------------
######用户和时间关系
u1=data[['user_id','hour']]
u1['user_hour']=1
u1=u1.groupby(['user_id','hour']).agg('sum').reset_index()
user_hour=u1
#####用户和查询类的关系
u1=data[['user_id','predict_category_property_m1']]
u1['user_pre_cate_m1']=1
u1=u1.groupby(['user_id','predict_category_property_m1']).agg('sum').reset_index()
user_pre_cate_m1=u1
####用户和page关系
u1=data[['user_id','context_page_id']]
u1['user_page']=1
u1=u1.groupby(['user_id','context_page_id']).agg('sum').reset_index()
user_page=u1


#########-------------------用户和店铺的特征--------------
#####用户和shop关系
u1=data[['user_id','shop_id']]
u1['user_shop']=1
u1=u1.groupby(['user_id','shop_id']).agg('sum').reset_index()
user_shop=u1


#######---a---------------商品和店铺的交叉
####商品和shopid浏览
u1=data[['item_id','shop_id']]
u1['item_shop']=1
u1=u1.groupby(['item_id','shop_id']).agg('sum').reset_index()
item_shop=u1

##########cateogry  and  ahop id
u1=data[['item_category_list','shop_id']]
u1['item_cate_shop']=1
u1=u1.groupby(['item_category_list','shop_id']).agg('sum').reset_index()
item_cate_shop=u1

#############  item_property m1  and shop
u1=data[['item_property_list_m1','shop_id']]
u1['item_pro_m1']=1
u1=u1.groupby(['item_property_list_m1','shop_id']).agg('sum').reset_index()
item_pro_m1=u1
#############  item_property m1  and shop
u1=data[['item_property_list_m2','shop_id']]
u1['item_pro_m2']=1
u1=u1.groupby(['item_property_list_m2','shop_id']).agg('sum').reset_index()
item_pro_m2=u1
#############  item_property m1  and shop
u1=data[['item_property_list_m3','shop_id']]
u1['item_pro_m3']=1
u1=u1.groupby(['item_property_list_m3','shop_id']).agg('sum').reset_index()
item_pro_m3=u1

#####iten brand  and shop
u1=data[['item_brand_id','shop_id']]
u1['item_brand_shop']=1
u1=u1.groupby(['item_brand_id','shop_id']).agg('sum').reset_index()
item_brand_shop=u1

#########---------------------商铺和上下文特征--------------
#####item  anf  page
u1=data[['item_id','context_page_id']]
u1['item_page']=1
u1=u1.groupby(['item_id','context_page_id']).agg('sum').reset_index()
item_page=u1

##3###item and pre_pro
u1=data[['item_id','predict_category_property_m1']]
u1['item_pre_cate']=1
u1=u1.groupby(['item_id','predict_category_property_m1']).agg('sum').reset_index()
item_pre_cate=u1

#####cate and pre_cate
u1=data[['item_category_list','predict_category_property_m1']]
u1['item_cate_proper']=1
u1=u1.groupby(['item_category_list','predict_category_property_m1']).agg('sum').reset_index()
item_cate_proper=u1
######item and hour
u1=data[['item_id','hour']]
u1['item_hour']=1
u1=u1.groupby(['item_id','hour']).agg('sum').reset_index()
item_hour=u1

###########################################item_brand  and pre pro
u1=data[['item_brand_id','predict_category_property_m1']]
u1['item_brand_pre_cate']=1
u1=u1.groupby(['item_brand_id','predict_category_property_m1']).agg('sum').reset_index()
item_brand_pre_cate=u1


###########---------------------商铺和上下文交叉----------------------
u1=data[['shop_id','hour']]
u1['shop_hour']=1
u1=u1.groupby(['shop_id','hour']).agg('sum').reset_index()
shop_hour=u1

##########商铺和pre cate
u1=data[['shop_id','predict_category_property_m1']]
u1['shop_pre_cate']=1
u1=u1.groupby(['shop_id','predict_category_property_m1']).agg('sum').reset_index()
shop_pre_cate=u1


all_two_data=data

all_two_data=pd.merge(all_two_data,item_brand_pre_cate,on=['item_brand_id','predict_category_property_m1'],how='left')
del item_brand_pre_cate
all_two_data=pd.merge(all_two_data,user_item,on=['user_id','item_id'],how='left')
del user_item
all_two_data=pd.merge(all_two_data,user_item_cate,on=['user_id','item_category_list'],how='left')
del user_item_cate
all_two_data=pd.merge(all_two_data,user_item_pro_m1,on=['user_id','item_property_list_m1'],how='left')
del user_item_pro_m1
all_two_data=pd.merge(all_two_data,user_item_pro_m2,on=['user_id','item_property_list_m2'],how='left')
del user_item_pro_m2
all_two_data=pd.merge(all_two_data,user_item_pro_m3,on=['user_id','item_property_list_m3'],how='left')
del user_item_pro_m3
all_two_data=pd.merge(all_two_data,user_item_brand,on=['user_id','item_brand_id'],how='left')
del user_item_brand
all_two_data=pd.merge(all_two_data,user_hour,on=['user_id','hour'],how='left')
del user_hour
all_two_data=pd.merge(all_two_data,user_pre_cate_m1,on=['user_id','predict_category_property_m1'],how='left')
del user_pre_cate_m1
all_two_data=pd.merge(all_two_data,user_page,on=['user_id','context_page_id'],how='left')
del user_page
all_two_data=pd.merge(all_two_data,user_shop,on=['user_id','shop_id'],how='left')
del user_shop
all_two_data=pd.merge(all_two_data,item_shop,on=['item_id','shop_id'],how='left')
del item_shop
all_two_data=pd.merge(all_two_data,item_cate_shop,on=['item_category_list','shop_id'],how='left')
del item_cate_shop
all_two_data=pd.merge(all_two_data,item_pro_m1,on=['item_property_list_m1','shop_id'],how='left')
del item_pro_m1
all_two_data=pd.merge(all_two_data,item_pro_m2,on=['item_property_list_m2','shop_id'],how='left')
del item_pro_m2
all_two_data=pd.merge(all_two_data,item_pro_m3,on=['item_property_list_m3','shop_id'],how='left')
del item_pro_m3
all_two_data=pd.merge(all_two_data,item_brand_shop,on=['item_brand_id','shop_id'],how='left')
del item_brand_shop
all_two_data=pd.merge(all_two_data,item_page,on=['item_id','context_page_id'],how='left')
del item_page
all_two_data=pd.merge(all_two_data,item_pre_cate,on=['item_id','predict_category_property_m1'],how='left')
del item_pre_cate
all_two_data=pd.merge(all_two_data,item_cate_proper,on=['item_category_list','predict_category_property_m1'],how='left')
del item_cate_proper
all_two_data=pd.merge(all_two_data,item_hour,on=['item_id','hour'],how='left')
del item_hour
all_two_data=pd.merge(all_two_data,shop_hour,on=['shop_id','hour'],how='left')
del shop_hour
all_two_data=pd.merge(all_two_data,shop_pre_cate,on=['shop_id','predict_category_property_m1'],how='left')
del shop_pre_cate


all_two_data=pd.merge(all_two_data,user_item_price,on=['user_id','item_price_level'],how='left')
del user_item_price
all_two_data=pd.merge(all_two_data,user_item_sales,on=['user_id','item_sales_level'],how='left')
del user_item_sales
all_two_data=pd.merge(all_two_data,user_item_collected,on=['user_id','item_collected_level'],how='left')
del user_item_collected
all_two_data=pd.merge(all_two_data,user_item_pv,on=['user_id','item_pv_level'],how='left')
del user_item_pv
all_two_data=pd.merge(all_two_data,user_item_city,on=['user_id','item_city_id'],how='left')
del user_item_city

all_two_data=all_two_data.fillna(0)
#all_two_data.to_csv('all_two_data.csv',index=None)
string_column=data.columns.values.tolist()
for ss in string_column:
    del all_two_data[ss]

##############----------------------------------用户购买商品的三级交叉特征-------------------------------------
#########user and cate and pro_m1
u1=data[['user_id','item_category_list','item_property_list_m1']]
u1['user_item_cate_pro_m1']=1
u1=u1.groupby(['user_id','item_category_list','item_property_list_m1']).agg('sum').reset_index()
user_item_cate_pro_m1=u1

#########user and cate and pro_m2
u1=data[['user_id','item_category_list','item_property_list_m2']]
u1['user_item_cate_pro_m2']=1
u1=u1.groupby(['user_id','item_category_list','item_property_list_m2']).agg('sum').reset_index()
user_item_cate_pro_m2=u1

#########user and cate and pro_m3
u1=data[['user_id','item_category_list','item_property_list_m3']]
u1['user_item_cate_pro_m3']=1
u1=u1.groupby(['user_id','item_category_list','item_property_list_m3']).agg('sum').reset_index()
user_item_cate_pro_m3=u1

###########user and item_brand and cate
u1=data[['user_id','item_brand_id','item_category_list']]
u1['user_brand_cate']=1
u1=u1.groupby(['user_id','item_brand_id','item_category_list']).agg('sum').reset_index()
user_brand_cate=u1
##########user brand and pro m1
u1=data[['user_id','item_brand_id','item_property_list_m1']]
u1['user_brand_pro_m1']=1
u1=u1.groupby(['user_id','item_brand_id','item_property_list_m1']).agg('sum').reset_index()
user_brand_pro_m1=u1
##########user brand and pro m2
u1=data[['user_id','item_brand_id','item_property_list_m2']]
u1['user_brand_pro_m2']=1
u1=u1.groupby(['user_id','item_brand_id','item_property_list_m2']).agg('sum').reset_index()
user_brand_pro_m2=u1
##########user brand and pro m3
u1=data[['user_id','item_brand_id','item_property_list_m3']]
u1['user_brand_pro_m3']=1
u1=u1.groupby(['user_id','item_brand_id','item_property_list_m3']).agg('sum').reset_index()
user_brand_pro_m3=u1
#############user shop and cate
u1=data[['user_id','shop_id','item_category_list']]
u1['user_shop_cate']=1
u1=u1.groupby(['user_id','shop_id','item_category_list']).agg('sum').reset_index()
user_shop_cate=u1
#############usr and shop and brand
u1=data[['user_id','shop_id','item_brand_id']]
u1['user_shop_brand']=1
u1=u1.groupby(['user_id','shop_id','item_brand_id']).agg('sum').reset_index()
user_shop_brand=u1
#############user and shop_pro+m1
u1=data[['user_id','shop_id','item_property_list_m1']]
u1['user_shop_pro_m1']=1
u1=u1.groupby(['user_id','shop_id','item_property_list_m1']).agg('sum').reset_index()
user_shop_pro_m1=u1
#############user and shop_pro+m2
u1=data[['user_id','shop_id','item_property_list_m2']]
u1['user_shop_pro_m2']=1
u1=u1.groupby(['user_id','shop_id','item_property_list_m2']).agg('sum').reset_index()
user_shop_pro_m2=u1
#############user and shop_pro+m3
u1=data[['user_id','shop_id','item_property_list_m3']]
u1['user_shop_pro_m3']=1
u1=u1.groupby(['user_id','shop_id','item_property_list_m3']).agg('sum').reset_index()
user_shop_pro_m3=u1



all_three_data=data

all_three_data=pd.merge(all_three_data,user_item_cate_pro_m1,on=['user_id','item_category_list','item_property_list_m1'],how='left')
del user_item_cate_pro_m1
all_three_data=pd.merge(all_three_data,user_item_cate_pro_m2,on=['user_id','item_category_list','item_property_list_m2'],how='left')
del user_item_cate_pro_m2
all_three_data=pd.merge(all_three_data,user_item_cate_pro_m3,on=['user_id','item_category_list','item_property_list_m3'],how='left')
del user_item_cate_pro_m3
all_three_data=pd.merge(all_three_data,user_brand_cate,on=['user_id','item_brand_id','item_category_list'],how='left')
del user_brand_cate
all_three_data=pd.merge(all_three_data,user_brand_pro_m1,on=['user_id','item_brand_id','item_property_list_m1'],how='left')
del user_brand_pro_m1
all_three_data=pd.merge(all_three_data,user_brand_pro_m2,on=['user_id','item_brand_id','item_property_list_m2'],how='left')
del user_brand_pro_m2
all_three_data=pd.merge(all_three_data,user_brand_pro_m3,on=['user_id','item_brand_id','item_property_list_m3'],how='left')
del user_brand_pro_m3
all_three_data=pd.merge(all_three_data,user_shop_cate,on=['user_id','shop_id','item_category_list'],how='left')
del user_shop_cate
all_three_data=pd.merge(all_three_data,user_shop_brand,on=['user_id','shop_id','item_brand_id'],how='left')
del user_shop_brand
all_three_data=pd.merge(all_three_data,user_shop_pro_m1,on=['user_id','shop_id','item_property_list_m1'],how='left')
del user_shop_pro_m1
all_three_data=pd.merge(all_three_data,user_shop_pro_m2,on=['user_id','shop_id','item_property_list_m2'],how='left')
del user_shop_pro_m2
all_three_data=pd.merge(all_three_data,user_shop_pro_m3,on=['user_id','shop_id','item_property_list_m3'],how='left')
del user_shop_pro_m3

all_three_data=all_three_data.fillna(0)

string_column=data.columns.values.tolist()

for ss in string_column:
    del all_three_data[ss]


###########------------------关于各种操作的时间以及累加次数的统计-------------------

######用户操作距离上一条的时间差

u1=data[['user_id','context_timestamp']].sort_values(by=['user_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)

u1['user_shift']=u1['user_id'].shift(1)
u1['time_delta_user']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[u1.user_shift!=u1.user_id,'time_delta_user']=-1
u1=u1.drop('user_shift',axis=1)
time_delta_user=u1

######商品id上一次浏览时间差
u1=data[['item_brand_id','context_timestamp']].sort_values(by=['item_brand_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['item_brand_shift']=u1['item_brand_id'].shift(1)
u1['time_delta_item_brand']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[u1.item_brand_shift!=u1.item_brand_id,'time_delta_item_brand']=-1
u1=u1.drop('item_brand_shift',axis=1)
time_delta_item_brand=u1


######商品不同品牌上一次浏览时间差
u1=data[['item_id','context_timestamp']].sort_values(by=['item_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['item_shift']=u1['item_id'].shift(1)
u1['time_delta_item']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[u1.item_shift!=u1.item_id,'time_delta_item']=-1
u1=u1.drop('item_shift',axis=1)
time_delta_item=u1

######店铺距离上一次浏览的时间差
u1=data[['shop_id','context_timestamp']].sort_values(by=['shop_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['shop_shift']=u1['shop_id'].shift(1)
u1['time_delta_shop']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[u1.shop_shift!=u1.shop_id,'time_delta_shop']=-1
u1=u1.drop('shop_shift',axis=1)
time_delta_shop=u1

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

#######用户浏览商品品牌的时间差
u1=data[['user_id','item_brand_id','context_timestamp']].sort_values(by=['user_id','item_brand_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']= u1['user_id'].shift(1)
u1['brand_shift'] = u1['item_brand_id'].shift(1)
u1['time_delta_user_brand'] = u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.brand_shift !=u1.item_brand_id),'time_delta_user_brand']=-1
u1=u1.drop(['user_shift','brand_shift'],axis=1)
time_delta_user_brand=u1

######用户浏览店铺时间差
u1=data[['user_id','shop_id','context_timestamp']].sort_values(by=['user_id','shop_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['shop_shift']=u1['shop_id'].shift(1)
u1['time_delta_user_shop']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_shift !=u1.shop_id),'time_delta_user_shop']=-1
u1=u1.drop(['user_shift','shop_shift'],axis=1)
time_delta_user_shop=u1

######用户浏览category时间差
u1=data[['user_id','item_category_list','context_timestamp']].sort_values(by=['user_id','item_category_list','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['category_shift']=u1['item_category_list'].shift(1)
u1['time_delta_user_cate']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.category_shift !=u1.item_category_list),'time_delta_user_cate']=-1
u1=u1.drop(['user_shift','category_shift'],axis=1)
time_delta_user_cate=u1



##############################用户浏览shop和商品种类的时间差
u1=data[['user_id','shop_id','item_category_list','context_timestamp']].sort_values(by=['user_id','shop_id','item_category_list','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['shop_id_shift']=u1['shop_id'].shift(1)
u1['item_category_list_shift']=u1['item_category_list'].shift(1)
u1['time_user_shop_item_cate_list']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_category_list_shift !=u1.item_category_list),'time_user_shop_item_cate_list']=-1
u1=u1.drop(['user_shift','shop_id_shift','item_category_list_shift'],axis=1)
time_user_shop_item_cate_list=u1
#############################用户浏览shop和brand的时间差
u1=data[['user_id','shop_id','item_brand_id','context_timestamp']].sort_values(by=['user_id','shop_id','item_brand_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['shop_id_shift']=u1['shop_id'].shift(1)
u1['item_brand_id_shift']=u1['item_brand_id'].shift(1)
u1['time_user_shop_brand']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_brand_id_shift !=u1.item_brand_id),'time_user_shop_brand']=-1
u1=u1.drop(['user_shift','shop_id_shift','item_brand_id_shift'],axis=1)
time_user_shop_brand=u1
################################用户浏览shop  和item pro list
u1=data[['user_id','shop_id','item_property_list_m1','context_timestamp']].sort_values(by=['user_id','shop_id','item_property_list_m1','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['shop_id_shift']=u1['shop_id'].shift(1)
u1['item_propert_list_m1_shift']=u1['item_property_list_m1'].shift(1)
u1['time_user_shop_item_pro_m1']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_propert_list_m1_shift !=u1.item_property_list_m1),'time_user_shop_item_pro_m1']=-1
u1=u1.drop(['user_shift','shop_id_shift','item_propert_list_m1_shift'],axis=1)
time_user_shop_item_pro_m1=u1
################################用户浏览shop  和item pro list
u1=data[['user_id','shop_id','item_property_list_m2','context_timestamp']].sort_values(by=['user_id','shop_id','item_property_list_m2','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['shop_id_shift']=u1['shop_id'].shift(1)
u1['item_propert_list_m2_shift']=u1['item_property_list_m2'].shift(1)
u1['time_user_shop_item_pro_m2']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_propert_list_m2_shift !=u1.item_property_list_m2),'time_user_shop_item_pro_m2']=-1
u1=u1.drop(['user_shift','shop_id_shift','item_propert_list_m2_shift'],axis=1)
time_user_shop_item_pro_m2=u1
################################用户浏览shop  和item pro list
u1=data[['user_id','shop_id','item_property_list_m3','context_timestamp']].sort_values(by=['user_id','shop_id','item_property_list_m3','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['shop_id_shift']=u1['shop_id'].shift(1)
u1['item_propert_list_m3_shift']=u1['item_property_list_m3'].shift(1)
u1['time_user_shop_item_pro_m3']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_propert_list_m3_shift !=u1.item_property_list_m3),'time_user_shop_item_pro_m3']=-1
u1=u1.drop(['user_shift','shop_id_shift','item_propert_list_m3_shift'],axis=1)
time_user_shop_item_pro_m3=u1
#


#
##############用户浏览cate and pro_m1的时间差
u1=data[['user_id','item_category_list','item_property_list_m1','context_timestamp']].sort_values(by=['user_id','item_category_list','item_property_list_m1','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['item_category_list_shift']=u1['item_category_list'].shift(1)
u1['item_propert_list_m1_shift']=u1['item_property_list_m1'].shift(1)
u1['time_user_cate_pro_m1']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_category_list_shift !=u1.item_category_list)|(u1.item_propert_list_m1_shift !=u1.item_property_list_m1),'time_user_cate_pro_m1']=-1
u1=u1.drop(['user_shift','item_category_list_shift','item_propert_list_m1_shift'],axis=1)
time_user_cate_pro_m1=u1
##############用户浏览cate and pro_m2的时间差
u1=data[['user_id','item_category_list','item_property_list_m2','context_timestamp']].sort_values(by=['user_id','item_category_list','item_property_list_m2','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['item_category_list_shift']=u1['item_category_list'].shift(1)
u1['item_propert_list_m2_shift']=u1['item_property_list_m2'].shift(1)
u1['time_user_cate_pro_m2']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_category_list_shift !=u1.item_category_list)|(u1.item_propert_list_m2_shift !=u1.item_property_list_m2),'time_user_cate_pro_m2']=-1
u1=u1.drop(['user_shift','item_category_list_shift','item_propert_list_m2_shift'],axis=1)
time_user_cate_pro_m2=u1
##############用户浏览cate and pro_m3的时间差
u1=data[['user_id','item_category_list','item_property_list_m3','context_timestamp']].sort_values(by=['user_id','item_category_list','item_property_list_m3','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(1)
u1['item_category_list_shift']=u1['item_category_list'].shift(1)
u1['item_propert_list_m3_shift']=u1['item_property_list_m3'].shift(1)
u1['time_user_cate_pro_m3']=u1['context_timestamp']-u1['context_timestamp'].shift(1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_category_list_shift !=u1.item_category_list)|(u1.item_propert_list_m3_shift !=u1.item_property_list_m3),'time_user_cate_pro_m3']=-1
u1=u1.drop(['user_shift','item_category_list_shift','item_propert_list_m3_shift'],axis=1)
time_user_cate_pro_m3=u1



all_time_data=data

all_time_data=pd.merge(all_time_data,time_delta_user,on=['user_id','context_timestamp'],how='left')
del time_delta_user
all_time_data=pd.merge(all_time_data,time_delta_item,on=['item_id','context_timestamp'],how='left')
del time_delta_item
all_time_data=pd.merge(all_time_data,time_delta_item_brand,on=['item_brand_id','context_timestamp'],how='left')
del time_delta_item_brand
all_time_data=pd.merge(all_time_data,time_delta_shop,on=['shop_id','context_timestamp'],how='left')
del time_delta_shop
all_time_data=pd.merge(all_time_data,time_delta_user_item,on=['user_id','item_id','context_timestamp'],how='left')
del time_delta_user_item
all_time_data=pd.merge(all_time_data,time_delta_user_brand,on=['user_id','item_brand_id','context_timestamp'],how='left')
del time_delta_user_brand
all_time_data=pd.merge(all_time_data,time_delta_user_shop,on=['user_id','shop_id','context_timestamp'],how='left')
del time_delta_user_shop

all_time_data=pd.merge(all_time_data,time_delta_user_cate,on=['user_id','item_category_list','context_timestamp'],how='left')
del time_delta_user_cate
all_time_data=pd.merge(all_time_data,time_user_cate_pro_m1,on=['user_id','item_category_list','item_property_list_m1','context_timestamp'],how='left')
del time_user_cate_pro_m1
all_time_data=pd.merge(all_time_data,time_user_cate_pro_m2,on=['user_id','item_category_list','item_property_list_m2','context_timestamp'],how='left')
del time_user_cate_pro_m2
all_time_data=pd.merge(all_time_data,time_user_cate_pro_m3,on=['user_id','item_category_list','item_property_list_m3','context_timestamp'],how='left')
del time_user_cate_pro_m3


all_time_data=pd.merge(all_time_data,time_user_shop_item_cate_list,on=['user_id','shop_id','item_category_list','context_timestamp'],how='left')
del time_user_shop_item_cate_list
all_time_data=pd.merge(all_time_data,time_user_shop_brand,on=['user_id','shop_id','item_brand_id','context_timestamp'],how='left')
del time_user_shop_brand
all_time_data=pd.merge(all_time_data,time_user_shop_item_pro_m1,on=['user_id','shop_id','item_property_list_m1','context_timestamp'],how='left')
del time_user_shop_item_pro_m1
all_time_data=pd.merge(all_time_data,time_user_shop_item_pro_m2,on=['user_id','shop_id','item_property_list_m2','context_timestamp'],how='left')
del time_user_shop_item_pro_m2
all_time_data=pd.merge(all_time_data,time_user_shop_item_pro_m3,on=['user_id','shop_id','item_property_list_m3','context_timestamp'],how='left')
del time_user_shop_item_pro_m3



##########-----------------------------------------持此浏览距离下一次浏览的时间差------------------------------------

#####用户操作距离上一条的时间差

u1=data[['user_id','context_timestamp']].sort_values(by=['user_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)

u1['user_shift']=u1['user_id'].shift(-1)
u1['before_time_delta_user']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[u1.user_shift!=u1.user_id,'before_time_delta_user']=1
u1=u1.drop('user_shift',axis=1)
before_time_delta_user=u1

######商品id上一次浏览时间差
u1=data[['item_brand_id','context_timestamp']].sort_values(by=['item_brand_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['item_brand_shift']=u1['item_brand_id'].shift(-1)
u1['before_time_delta_item_brand']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[u1.item_brand_shift!=u1.item_brand_id,'before_time_delta_item_brand']=1
u1=u1.drop('item_brand_shift',axis=1)
before_time_delta_item_brand=u1


######商品不同品牌上一次浏览时间差
u1=data[['item_id','context_timestamp']].sort_values(by=['item_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['item_shift']=u1['item_id'].shift(-1)
u1['before_time_delta_item']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[u1.item_shift!=u1.item_id,'before_time_delta_item']=1
u1=u1.drop('item_shift',axis=1)
before_time_delta_item=u1

######店铺距离上一次浏览的时间差
u1=data[['shop_id','context_timestamp']].sort_values(by=['shop_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['shop_shift']=u1['shop_id'].shift(-1)
u1['before_time_delta_shop']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[u1.shop_shift!=u1.shop_id,'before_time_delta_shop']=1
u1=u1.drop('shop_shift',axis=1)
before_time_delta_shop=u1

#######用户购买商品编号之间的时间差
u1=data[['user_id','item_id','context_timestamp']].sort_values(by=['user_id','item_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift'] = u1['user_id'].shift(-1)
u1['item_shift'] = u1['item_id'].shift(-1)
u1['before_time_delta_user_item'] = u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_shift !=u1.item_id),'before_time_delta_user_item']=1
u1=u1.drop(['user_shift','item_shift'],axis=1)
before_time_delta_user_item=u1

#######用户浏览商品品牌的时间差
u1=data[['user_id','item_brand_id','context_timestamp']].sort_values(by=['user_id','item_brand_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']= u1['user_id'].shift(-1)
u1['brand_shift'] = u1['item_brand_id'].shift(-1)
u1['before_time_delta_user_brand'] = u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.brand_shift !=u1.item_brand_id),'before_time_delta_user_brand']=1
u1=u1.drop(['user_shift','brand_shift'],axis=1)
before_time_delta_user_brand=u1

######用户浏览店铺时间差
u1=data[['user_id','shop_id','context_timestamp']].sort_values(by=['user_id','shop_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['shop_shift']=u1['shop_id'].shift(-1)
u1['before_time_delta_user_shop']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_shift !=u1.shop_id),'before_time_delta_user_shop']=1
u1=u1.drop(['user_shift','shop_shift'],axis=1)
before_time_delta_user_shop=u1

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
##############用户浏览cate and pro_m2的时间差
u1=data[['user_id','item_category_list','item_property_list_m2','context_timestamp']].sort_values(by=['user_id','item_category_list','item_property_list_m2','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['item_category_list_shift']=u1['item_category_list'].shift(-1)
u1['item_propert_list_m2_shift']=u1['item_property_list_m2'].shift(-1)
u1['before_time_user_cate_pro_m2']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_category_list_shift !=u1.item_category_list)|(u1.item_propert_list_m2_shift !=u1.item_property_list_m2),'before_time_user_cate_pro_m2']=1
u1=u1.drop(['user_shift','item_category_list_shift','item_propert_list_m2_shift'],axis=1)
before_time_user_cate_pro_m2=u1
##############用户浏览cate and pro_m3的时间差
u1=data[['user_id','item_category_list','item_property_list_m3','context_timestamp']].sort_values(by=['user_id','item_category_list','item_property_list_m3','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['item_category_list_shift']=u1['item_category_list'].shift(-1)
u1['item_propert_list_m3_shift']=u1['item_property_list_m3'].shift(-1)
u1['before_time_user_cate_pro_m3']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.item_category_list_shift !=u1.item_category_list)|(u1.item_propert_list_m3_shift !=u1.item_property_list_m3),'before_time_user_cate_pro_m3']=1
u1=u1.drop(['user_shift','item_category_list_shift','item_propert_list_m3_shift'],axis=1)
before_time_user_cate_pro_m3=u1


##############################用户浏览shop和商品种类的时间差
u1=data[['user_id','shop_id','item_category_list','context_timestamp']].sort_values(by=['user_id','shop_id','item_category_list','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['shop_id_shift']=u1['shop_id'].shift(-1)
u1['item_category_list_shift']=u1['item_category_list'].shift(-1)
u1['before_time_user_shop_item_cate_list']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_category_list_shift !=u1.item_category_list),'before_time_user_shop_item_cate_list']=1
u1=u1.drop(['user_shift','shop_id_shift','item_category_list_shift'],axis=1)
before_time_user_shop_item_cate_list=u1
#############################用户浏览shop和brand的时间差
u1=data[['user_id','shop_id','item_brand_id','context_timestamp']].sort_values(by=['user_id','shop_id','item_brand_id','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['shop_id_shift']=u1['shop_id'].shift(-1)
u1['item_brand_id_shift']=u1['item_brand_id'].shift(-1)
u1['before_time_user_shop_brand']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_brand_id_shift !=u1.item_brand_id),'before_time_user_shop_brand']=1
u1=u1.drop(['user_shift','shop_id_shift','item_brand_id_shift'],axis=1)
before_time_user_shop_brand=u1
################################用户浏览shop  和item pro list
u1=data[['user_id','shop_id','item_property_list_m1','context_timestamp']].sort_values(by=['user_id','shop_id','item_property_list_m1','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['shop_id_shift']=u1['shop_id'].shift(-1)
u1['item_propert_list_m1_shift']=u1['item_property_list_m1'].shift(-1)
u1['before_time_user_shop_item_pro_m1']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_propert_list_m1_shift !=u1.item_property_list_m1),'before_time_user_shop_item_pro_m1']=1
u1=u1.drop(['user_shift','shop_id_shift','item_propert_list_m1_shift'],axis=1)
before_time_user_shop_item_pro_m1=u1

################################用户浏览shop  和item pro list
u1=data[['user_id','shop_id','item_property_list_m2','context_timestamp']].sort_values(by=['user_id','shop_id','item_property_list_m2','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['shop_id_shift']=u1['shop_id'].shift(-1)
u1['item_propert_list_m2_shift']=u1['item_property_list_m2'].shift(-1)
u1['before_time_user_shop_item_pro_m2']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_propert_list_m2_shift !=u1.item_property_list_m2),'before_time_user_shop_item_pro_m2']=1
u1=u1.drop(['user_shift','shop_id_shift','item_propert_list_m2_shift'],axis=1)
before_time_user_shop_item_pro_m2=u1
################################用户浏览shop  和item pro list
u1=data[['user_id','shop_id','item_property_list_m3','context_timestamp']].sort_values(by=['user_id','shop_id','item_property_list_m3','context_timestamp']).reset_index()
del u1['index']
u1.drop_duplicates(inplace=True)
u1['user_shift']=u1['user_id'].shift(-1)
u1['shop_id_shift']=u1['shop_id'].shift(-1)
u1['item_propert_list_m3_shift']=u1['item_property_list_m3'].shift(-1)
u1['before_time_user_shop_item_pro_m3']=u1['context_timestamp']-u1['context_timestamp'].shift(-1)
u1.loc[(u1.user_shift !=u1.user_id)|(u1.shop_id_shift !=u1.shop_id)|(u1.item_propert_list_m3_shift !=u1.item_property_list_m3),'before_time_user_shop_item_pro_m3']=1
u1=u1.drop(['user_shift','shop_id_shift','item_propert_list_m3_shift'],axis=1)
before_time_user_shop_item_pro_m3=u1



all_time_data=pd.merge(all_time_data,before_time_delta_user,on=['user_id','context_timestamp'],how='left')
del before_time_delta_user
all_time_data=pd.merge(all_time_data,before_time_delta_item,on=['item_id','context_timestamp'],how='left')
del before_time_delta_item
all_time_data=pd.merge(all_time_data,before_time_delta_item_brand,on=['item_brand_id','context_timestamp'],how='left')
del before_time_delta_item_brand
all_time_data=pd.merge(all_time_data,before_time_delta_shop,on=['shop_id','context_timestamp'],how='left')
del before_time_delta_shop
all_time_data=pd.merge(all_time_data,before_time_delta_user_item,on=['user_id','item_id','context_timestamp'],how='left')
del before_time_delta_user_item
all_time_data=pd.merge(all_time_data,before_time_delta_user_brand,on=['user_id','item_brand_id','context_timestamp'],how='left')
del before_time_delta_user_brand
all_time_data=pd.merge(all_time_data,before_time_delta_user_shop,on=['user_id','shop_id','context_timestamp'],how='left')
del before_time_delta_user_shop

all_time_data=pd.merge(all_time_data,before_time_delta_user_cate,on=['user_id','item_category_list','context_timestamp'],how='left')
del before_time_delta_user_cate

all_time_data.merge_time_delta_user=all_time_data.before_time_delta_user+all_time_data.time_delta_user
all_time_data.merge_time_delta_item=all_time_data.before_time_delta_item+all_time_data.time_delta_item
all_time_data.megre_time_delta_item_brand=all_time_data.before_time_delta_item_brand+all_time_data.time_delta_item_brand
all_time_data.megre_time_delta_shop=all_time_data.before_time_delta_user_item+all_time_data.time_delta_user_item
all_time_data.megre_time_delta_user_brand=all_time_data.before_time_delta_user_brand+all_time_data.time_delta_user_brand
all_time_data.megre_time_delta_user_shop=all_time_data.before_time_delta_user_shop+all_time_data.time_delta_user_shop


all_time_data=pd.merge(all_time_data,before_time_user_cate_pro_m1,on=['user_id','item_category_list','item_property_list_m1','context_timestamp'],how='left')
del before_time_user_cate_pro_m1
all_time_data=pd.merge(all_time_data,before_time_user_cate_pro_m2,on=['user_id','item_category_list','item_property_list_m2','context_timestamp'],how='left')
del before_time_user_cate_pro_m2
all_time_data=pd.merge(all_time_data,before_time_user_cate_pro_m3,on=['user_id','item_category_list','item_property_list_m3','context_timestamp'],how='left')
del before_time_user_cate_pro_m3


all_time_data=pd.merge(all_time_data,before_time_user_shop_item_cate_list,on=['user_id','shop_id','item_category_list','context_timestamp'],how='left')
del before_time_user_shop_item_cate_list
all_time_data=pd.merge(all_time_data,before_time_user_shop_brand,on=['user_id','shop_id','item_brand_id','context_timestamp'],how='left')
del before_time_user_shop_brand
all_time_data=pd.merge(all_time_data,before_time_user_shop_item_pro_m1,on=['user_id','shop_id','item_property_list_m1','context_timestamp'],how='left')
del before_time_user_shop_item_pro_m1

all_time_data=pd.merge(all_time_data,before_time_user_shop_item_pro_m2,on=['user_id','shop_id','item_property_list_m2','context_timestamp'],how='left')
del before_time_user_shop_item_pro_m2
all_time_data=pd.merge(all_time_data,before_time_user_shop_item_pro_m3,on=['user_id','shop_id','item_property_list_m3','context_timestamp'],how='left')
del before_time_user_shop_item_pro_m3


##----------------------用户和商品以及商户的此前出现次数，当天出现的次数，当天出现的次数累加等-----------------------

#####用户和day：在一天内的用户的累加浏览次数
u1=data[['user_id','day']]
u1['user_day']=1
#user_day_sum=u1.groupby(['user_id','day']).agg('sum').reset_index()

#u1['user_cumsum']=u1.groupby(['user_id'])['user_day'].cumsum()##############用户的累加浏览次数
u1['user_day_cumsum']=u1.groupby(['user_id','day'])['user_day'].cumsum()##############用户每天累加次数
user_day_cumsum=u1.drop(['user_id','day','user_day'],axis=1)


#####item和day：在一天内的商品编号的累加浏览次数
u1=data[['item_id','day']]
u1['item_day_times']=1
#item_day_sum=u1.groupby(['item_id','day']).agg('sum').reset_index()

#u1['item_cumsum']=u1.groupby(['item_id'])['item_day_times'].cumsum()##############商品id的累加浏览次数
u1['item_day_cumsum']=u1.groupby(['item_id','day'])['item_day_times'].cumsum()##############商品每天累加次数
item_day_cumsum=u1.drop(['item_id','day','item_day_times'],axis=1)



#####shop和day：在一天内的商店的编号的累加浏览次数
u1=data[['shop_id','day']]
u1['shop_day_times']=1
#shop_day_sum=u1.groupby(['shop_id','day']).agg('sum').reset_index()

#u1['shop_cumsum']=u1.groupby(['shop_id'])['shop_day_times'].cumsum()##############商品id的累加浏览次数
u1['shop_day_cumsum']=u1.groupby(['shop_id','day'])['shop_day_times'].cumsum()##############商品每天累加次数
shop_day_cumsum=u1.drop(['shop_id','day','shop_day_times'],axis=1)


########uese and itemm在倚天的累加浏览次数
u1=data[['user_id','item_id','day']]
u1['user_item_day_times']=1
#user_item_day_sum=u1.groupby(['user_id','item_id','day']).agg('sum').reset_index()

#u1['user_item_cumsum']=u1.groupby(['user_id','item_id'])['user_item_day_times'].cumsum()
u1['user_item_day_cumsum']=u1.groupby(['user_id','item_id','day'])['user_item_day_times'].cumsum()

user_item_day_cumsum=u1.drop(['user_id','item_id','day','user_item_day_times'],axis=1)

#############user  and  shopid的累加浏览次数
u1=data[['user_id','shop_id','day']]
u1['user_shop_day_times']=1
user_shop_day_sum=u1.groupby(['user_id','shop_id','day']).agg('sum').reset_index()

#u1['user_shop_cumsum']=u1.groupby(['user_id','shop_id'])['user_shop_day_times'].cumsum()
u1['user_shop_day_cumsum']=u1.groupby(['user_id','shop_id','day'])['user_shop_day_times'].cumsum()
del u1['user_shop_day_times']
u1=pd.merge(u1,user_shop_day_sum,on=['user_id','shop_id','day'],how='left')
#u1['is_last_user_shop']=0
#u1.loc[(u1.user_shop_day_cumsum==u1.user_shop_day_times)&(u1.user_shop_day_times>1),'is_last_user_shop']=1

user_shop_day_cumsum=u1.drop(['user_id','shop_id','day'],axis=1)
#del u1['user_shop_day_times']
#user_shop_day_cumsum=u1

##########uesr and category
u1=data[['user_id','item_category_list','day']]
u1['user_category_day_times']=1
user_categoyy_day_sum=u1.groupby(['user_id','item_category_list','day']).agg('sum').reset_index()

#u1['user_category_cumsum']=u1.groupby(['user_id','item_category_list'])['user_category_day_times'].cumsum()
u1['user_category_day_cumsum']=u1.groupby(['user_id','item_category_list','day'])['user_category_day_times'].cumsum()
del u1['user_category_day_times']
u1=pd.merge(u1,user_categoyy_day_sum,on=['user_id','item_category_list','day'],how='left')
#u1['is_last_user_category']=0
#u1.loc[(u1.user_category_day_cumsum==u1.user_category_day_times)&(u1.user_category_day_times>1),'is_last_user_category']=1

user_category_day_cumsum=u1.drop(['user_id','item_category_list','day'],axis=1)
#user_category_day_cumsum=u1

##-----------------user and iten brand-------
u1=data[['user_id','item_brand_id','day']]
u1['user_brand_day_times']=1
#user_brand_day_sum=u1.groupby(['user_id','item_brand_id','day']).agg('sum').reset_index()
#
#u1['user_brand_cumsum']=u1.groupby(['user_id','item_brand_id'])['user_brand_day_times'].cumsum()
u1['user_brand_day_cumsum']=u1.groupby(['user_id','item_brand_id','day'])['user_brand_day_times'].cumsum()

user_brand_day_cumsum=u1.drop(['user_id','item_brand_id','day','user_brand_day_times'],axis=1)


##########item  and  shop的日累加及星期累加次数
u1=data[['item_id','shop_id','day']]
u1['item_shop_day_times']=1
#item_shop_day_sum=u1.groupby(['item_id','shop_id','day']).agg('sum').reset_index()
#
#u1['item_shop_cumsum']=u1.groupby(['item_id','shop_id'])['item_shop_day_times'].cumsum()
u1['item_shop_day_cumsum']=u1.groupby(['item_id','shop_id','day'])['item_shop_day_times'].cumsum()
item_shop_day_cumsum=u1.drop(['item_id','shop_id','day','item_shop_day_times'],axis=1)

#del u1['item_shop_day_times']
#item_shop_day_cumsum=u1
##########item-categoty_day
u1=data[['item_category_list','day']]
u1['item_category_day_times']=1
#item_category_day_sum=u1.groupby(['item_category_list','day']).agg('sum').reset_index()
#
#u1['item_category_cumsum']=u1.groupby(['item_category_list'])['item_category_day_times'].cumsum()
u1['item_category_day_cumsum']=u1.groupby(['item_category_list','day'])['item_category_day_times'].cumsum()

item_category_day_cumsum=u1.drop(['item_category_list','day','item_category_day_times'],axis=1)


#----------------------------------##################-------------------------------
########user and Prom1
u1=data[['user_id','item_property_list_m1','day']]
u1['user_pro_m1_day_times']=1
#user_pro_m1_day_sum=u1.groupby(['user_id','item_property_list_m1','day']).agg('sum').reset_index()
#
#u1['user_pro_m1_cumsum']=u1.groupby(['user_id','item_property_list_m1'])['user_pro_m1_day_times'].cumsum()
u1['user_pro_m1_day_cumsum']=u1.groupby(['user_id','item_property_list_m1','day'])['user_pro_m1_day_times'].cumsum()
user_pro_m1_day_cumsum=u1.drop(['user_id','item_property_list_m1','day','user_pro_m1_day_times'],axis=1)
#########user and Prom2
u1=data[['user_id','item_property_list_m2','day']]
u1['user_pro_m2_day_times']=1
#user_pro_m2_day_sum=u1.groupby(['user_id','item_property_list_m2','day']).agg('sum').reset_index()
#
#u1['user_pro_m2_cumsum']=u1.groupby(['user_id','item_property_list_m2'])['user_pro_m2_day_times'].cumsum()
u1['user_pro_m2_day_cumsum']=u1.groupby(['user_id','item_property_list_m2','day'])['user_pro_m2_day_times'].cumsum()
user_pro_m2_day_cumsum=u1.drop(['user_id','item_property_list_m2','day','user_pro_m2_day_times'],axis=1)
#########user and Prom3
u1=data[['user_id','item_property_list_m3','day']]
u1['user_pro_m3_day_times']=1
#user_pro_m3_day_sum=u1.groupby(['user_id','item_property_list_m3','day']).agg('sum').reset_index()
#
#u1['user_pro_m3_cumsum']=u1.groupby(['user_id','item_property_list_m3'])['user_pro_m3_day_times'].cumsum()
u1['user_pro_m3_day_cumsum']=u1.groupby(['user_id','item_property_list_m3','day'])['user_pro_m3_day_times'].cumsum()
user_pro_m3_day_cumsum=u1.drop(['user_id','item_property_list_m3','day','user_pro_m3_day_times'],axis=1)



#############user and day and hour
u1=data[['user_id','day','hour']]
u1['user_day_hour']=1
user_day_hour=u1.groupby(['user_id','day','hour']).agg('sum').reset_index()

u1['user_day_hour_cumsum']=u1.groupby(['user_id','day','hour'])['user_day_hour'].cumsum()
user_day_hour_cumsum=u1.drop(['user_id','day','hour','user_day_hour'],axis=1)
###########item and day and hour
u1=data[['item_id','day','hour']]
u1['item_day_hour']=1
item_day_hour=u1.groupby(['item_id','day','hour']).agg('sum').reset_index()

u1['item_day_hour_cumsum']=u1.groupby(['item_id','day','hour'])['item_day_hour'].cumsum()
item_day_hour_cumsum=u1.drop(['item_id','day','hour','item_day_hour'],axis=1)
###########shop_id and day and hour
u1=data[['shop_id','day','hour']]
u1['shop_day_hour']=1
shop_day_hour=u1.groupby(['shop_id','day','hour']).agg('sum').reset_index()

u1['shop_day_hour_cumsum']=u1.groupby(['shop_id','day','hour'])['shop_day_hour'].cumsum()
shop_day_hour_cumsum=u1.drop(['shop_id','day','hour','shop_day_hour'],axis=1)

######################uesr and itm and day and hour
u1=data[['user_id','item_id','day','hour']]
u1['user_item_day_hour']=1
user_item_day_hour=u1.groupby(['user_id','item_id','day','hour']).agg('sum').reset_index()

u1['user_item_day_hour_cumsum']=u1.groupby(['user_id','item_id','day','hour'])['user_item_day_hour'].cumsum()
user_item_day_hour_cumsum=u1.drop(['user_id','item_id','day','hour','user_item_day_hour'],axis=1)
############user shop_id and day and hour
u1=data[['user_id','shop_id','day','hour']]
u1['user_shop_day_hour']=1
user_shop_day_hour=u1.groupby(['user_id','shop_id','day','hour']).agg('sum').reset_index()

u1['user_shop_day_hour_cumsum']=u1.groupby(['user_id','shop_id','day','hour'])['user_shop_day_hour'].cumsum()
user_shop_day_hour_cumsum=u1.drop(['user_id','shop_id','day','hour','user_shop_day_hour'],axis=1)

all_time_data=pd.merge(all_time_data,user_day_hour,on=['user_id','day','hour'],how='left')
del user_day_hour
all_time_data=pd.concat([all_time_data,user_day_hour_cumsum],axis=1,join='outer')
del user_day_hour_cumsum
all_time_data=pd.merge(all_time_data,item_day_hour,on=['item_id','day','hour'],how='left')
del item_day_hour
all_time_data=pd.concat([all_time_data,item_day_hour_cumsum],axis=1,join='outer')
del item_day_hour_cumsum
all_time_data=pd.merge(all_time_data,shop_day_hour,on=['shop_id','day','hour'],how='left')
del shop_day_hour
all_time_data=pd.concat([all_time_data,shop_day_hour_cumsum],axis=1,join='outer')
del shop_day_hour_cumsum
all_time_data=pd.merge(all_time_data,user_item_day_hour,on=['user_id','item_id','day','hour'],how='left')
del user_item_day_hour
all_time_data=pd.concat([all_time_data,user_item_day_hour_cumsum],axis=1,join='outer')
del user_item_day_hour_cumsum
all_time_data=pd.merge(all_time_data,user_shop_day_hour,on=['user_id','shop_id','day','hour'],how='left')
del user_shop_day_hour
all_time_data=pd.concat([all_time_data,user_shop_day_hour_cumsum],axis=1,join='outer')
del user_shop_day_hour_cumsum

all_time_data['user--user-item-hour-cum']=all_time_data.user_day_hour_cumsum-all_time_data.user_item_day_hour_cumsum
all_time_data['user--user-shop-hour-cum']=all_time_data.user_day_hour_cumsum-all_time_data.user_shop_day_hour_cumsum
all_time_data['item--user-item-hour-cum']=all_time_data.item_day_hour_cumsum-all_time_data.user_item_day_hour_cumsum
all_time_data['shop--user-shop_hour-cum']=all_time_data.shop_day_hour_cumsum-all_time_data.user_shop_day_hour_cumsum



#all_time_data=pd.merge(all_time_data,user_day_sum,on=['user_id','day'],how='left')
#del user_day_sum
all_time_data=pd.concat([all_time_data,user_day_cumsum],axis=1,join='outer')
del user_day_cumsum
#all_time_data=pd.merge(all_time_data,item_day_sum,on=['item_id','day'],how='left')
#del item_day_sum
all_time_data=pd.concat([all_time_data,item_day_cumsum],axis=1,join='outer')
del item_day_cumsum

#all_time_data=pd.merge(all_time_data,item_category_day_sum,on=['item_category_list','day'],how='left')
#del item_category_day_sum
all_time_data=pd.concat([all_time_data,item_category_day_cumsum],axis=1,join='outer')
del item_category_day_cumsum

#all_time_data=pd.merge(all_time_data,shop_day_sum,on=['shop_id','day'],how='left')
#del shop_day_sum
all_time_data=pd.concat([all_time_data,shop_day_cumsum],axis=1,join='outer')
del shop_day_cumsum
#all_time_data=pd.merge(all_time_data,user_item_day_sum,on=['user_id','item_id','day'],how='left')
#del user_item_day_sum
all_time_data=pd.concat([all_time_data,user_item_day_cumsum],axis=1,join='outer')
del user_item_day_cumsum


all_time_data=pd.concat([all_time_data,user_shop_day_cumsum],axis=1,join='outer')
del user_shop_day_cumsum

#all_time_data=pd.merge(all_time_data,item_shop_day_sum,on=['item_id','shop_id','day'],how='left')
#del item_shop_day_sum
all_time_data=pd.concat([all_time_data,item_shop_day_cumsum],axis=1,join='outer')
del item_shop_day_cumsum

all_time_data=pd.concat([all_time_data,user_category_day_cumsum],axis=1,join='outer')
del user_category_day_cumsum

#all_time_data['is_cate_leak']=0
#all_time_data.loc[(all_time_data.user_category_day_cumsum>1)&(all_time_data.user_category_day_cumsum==all_time_data.item_category_day_times),'is_cate_leak']=1
#all_time_data['is_shop_leak']=0
#all_time_data.loc[(all_time_data.user_shop_day_cumsum>1)&(all_time_data.user_shop_day_cumsum==all_time_data.shop_day_times),'is_shop_leak']=1


#all_time_data=pd.merge(all_time_data,user_brand_day_sum,on=['user_id','item_brand_id','day'],how='left')
#del user_brand_day_sum
all_time_data=pd.concat([all_time_data,user_brand_day_cumsum],axis=1,join='outer')
del user_brand_day_cumsum

#all_time_data=pd.merge(all_time_data,user_pro_m1_day_sum,on=['user_id','item_property_list_m1','day'],how='left')
#del user_pro_m1_day_sum
#all_time_data=pd.merge(all_time_data,user_pro_m2_day_sum,on=['user_id','item_property_list_m2','day'],how='left')
#del user_pro_m2_day_sum
#all_time_data=pd.merge(all_time_data,user_pro_m3_day_sum,on=['user_id','item_property_list_m3','day'],how='left')
#del user_pro_m3_day_sum
all_time_data=pd.concat([all_time_data,user_pro_m1_day_cumsum],axis=1,join='outer')
del user_pro_m1_day_cumsum
all_time_data=pd.concat([all_time_data,user_pro_m2_day_cumsum],axis=1,join='outer')
del user_pro_m2_day_cumsum
all_time_data=pd.concat([all_time_data,user_pro_m3_day_cumsum],axis=1,join='outer')
del user_pro_m3_day_cumsum

del user_shop_day_sum
del user_categoyy_day_sum


strigng=all_time_data.columns.values.tolist()
all_time_data=all_time_data.fillna(0)
#all_time_data.to_csv('all_time_data.csv',index=None)

string_column=data.columns.values.tolist()
for ss in string_column:
    del all_time_data[ss]

############拼接
new_data=pd.concat([data,all_item_data],axis=1)
del all_item_data
new_data=pd.concat([new_data,all_user_data],axis=1)
del all_user_data

new_data=pd.concat([new_data,all_shop_data],axis=1)
del all_shop_data
new_data=pd.concat([new_data,all_two_data],axis=1)
del all_two_data
new_data=pd.concat([new_data,all_time_data],axis=1)
del all_time_data
new_data=pd.concat([new_data,all_three_data],axis=1)
del all_three_data


##########划分数据集
#train=tr_data[(tr_data.day<24)].drop(['day','is_trade'],axis=1,inplace=False)
#test=tr_data[(tr_data.day==24)].drop(['day','is_trade'],axis=1,inplace=False)
#y_train=tr_data[(tr_data.day<24)].is_trade
#y_test=tr_data[(tr_data.day==24)].is_trade


del new_data['context_timestamp']
yuce_data=new_data[new_data.is_trade==10]
tr_data=new_data[new_data.is_trade !=10]
#
del new_data

##import seaborn as sns
##import matplotlib.pyplot as plt
##plt.figure(figsize=(8,6),dpi=120)
##sns.countplot(y='item_category_list',hue='is_trade',data=tr_data)
#
#
######
#
#
#
#
############lgb_train交叉验证只用第7天的数据

################---------------lightGBM转换数据格式--------------------
import json
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
######划分训练与验证
trainxy,valxy=train_test_split(tr_data,test_size=0.1,random_state=2018,stratify=tr_data.is_trade)


X_train=trainxy.drop(['day','is_trade'],axis=1,inplace=False)
y_train=trainxy.is_trade

X_test=valxy.drop(['day','is_trade'],axis=1,inplace=False)
y_test=valxy.is_trade
#
#tr_string=X_train.columns.values.tolist()
string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_age_level','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']
######数据切换
##lgb_train=lgb.Dataset(X_train,y_train,free_raw_data=False)
##lgb_eval=lgb.Dataset(X_test,y_test,free_raw_data=False)
##
##params={
##        'boosting_type':'gbdt',
##        'objective':'binary',
##        'metric':'binary_logloss',
##        'num_leaves':15,
##        'learning_rate':0.05,
##        'feature_fraction':0.8,
##        'bagging_fraction':1,
##        'lambda_l1':1
##        }
##
##
###bst=lgb.cv(params,lgb_train,seed=2018,num_boost_round=10000,nfold=5,feature_name=tr_string,categorical_feature=string3,early_stopping_rounds=50,verbose_eval=True)
##bst=lgb.cv(params,lgb_train,seed=2018,num_boost_round=10000,nfold=5,early_stopping_rounds=50,verbose_eval=True)
##
##
##num_boost_round=len(bst['binary_logloss-mean'])
##print('best_num_boost:',num_boost_round,'   best_logloss:',bst['binary_logloss-mean'][num_boost_round-1])
##
##estimators=lgb.train(params,lgb_train,num_boost_round)
##
#####查看重要性
##importance=estimators.feature_importance()
##names=estimators.feature_name()
##feat_imp = pd.Series(importance,names).sort_values(ascending=False)
##print(feat_imp)
##print(feat_imp.shape)
##
##for ss in names:
##    if(feat_imp[ss]==0):
##        del X_train[ss]
##        del X_test[ss]
##    
##   
##
##
##ypred=estimators.predict(X_test)
##log_loss(y_test,ypred)
##
#######用eval验证
##lgb_model=lgb.train(params,lgb_train,valid_sets=lgb_eval,num_boost_round=10000,early_stopping_rounds=50)
##pre_off=lgb_model.predict(X_test,num_iteration=lgb.best_iteration)
##
##
########on_line 预测
##online_test=yuce_data.drop(['day','is_trade'],axis=1,inplace=False)
##pre_on=lgb.predict(online_test,num_iteration=lgb.best_iteration)
##
##bao_cun=data_test[['instance_id']]
##bao_cun['predicted_score']=pre_on
##bao_cun.to_csv('round2_result/xgb0178805.txt',index=False,sep=' ')
##
#
#
#
##############xgboost预测
import xgboost as xgb

string_ob=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']

for ss in string_ob:
    del X_train[ss]
    del X_test[ss]



del X_train['hour']
del X_test['hour']

dtrain1=xgb.DMatrix(X_train,y_train)
dval=xgb.DMatrix(X_test,y_test)
#dtest=xgb.DMatrix(online_test)
#dval_test=xgb.DMatrix(off_line_test)
params={
        'booster':'gbtree',
        'objective':'binary:logistic',
#        'gamma':0.1
        'eta':0.05,
        'max_depth':3,
        'lambda':3,
        'subsample':0.9,
        'min_child_weight':5,
        'colsample_bytree':0.8,
        'silent':0,
        'eval_metric':'logloss'
        }




cvres=xgb.cv(params,dtrain1,num_boost_round=10000,nfold=5,early_stopping_rounds=50,verbose_eval=True)

num_boost_round=len(cvres['test-logloss-mean'])
print('best_num_boost:',num_boost_round,'   best_logloss:',cvres['test-logloss-mean'][num_boost_round-1])

#
estimator=xgb.train(params,dtrain1,num_boost_round)


#
xgb_score=estimator.get_fscore()
xgb_score=sorted(xgb_score.items(), key=lambda x:x[1],reverse=True)
#
##xgb_score1=pd.DataFrame(xgb_score)
##
##
##import time
##start=time.clock()
##
##watchlist=[(dtrain1,'train'),(dval,'val')]
##mmodel=xgb.train(params,dtrain1,num_boost_round=20000,evals=watchlist,early_stopping_rounds=10)
##sss=mmodel.best_iteration
####
##elapsed=(time.clock()-start)
##print("time used;",elapsed)
##
##########线下
pres_offline=estimator.predict(dval,ntree_limit=estimator.best_iteration)
print(log_loss(y_test,pres_offline))###0.170531

pd.DataFrame(pres_offline).to_csv('ensamble/pre_off_xgb_170381.txt',index=False)

#####线上
online_test=yuce_data.drop(['day','is_trade'],axis=1,inplace=False)
for ss in string_ob:
    del online_test[ss]
#
dtest=xgb.DMatrix(online_test)
pre_online=estimator.predict(dtest,ntree_limit=estimator.best_iteration)

bao_cun=data_test[['instance_id']]
bao_cun['predicted_score']=pre_online
bao_cun.to_csv('round2_result/xgb0170381.txt',index=False,sep=' ')


##bao_cun=data_test[['instance_id']]
##bao_cun['predicted_score']=pre_online
##bao_cun.to_csv('round2_result/xgb0178805.txt',index=False,sep=' ')
#
##BAO_CHUN=pd.read_csv('xgb0180716.txt',delimiter=' ',header=0)
##BAO_CHUN2=pd.read_csv('round2_result/b201851_18315.txt',delimiter=' ',header=0)
##
##feature_score = mmodel.get_fscore()
##feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
##fs = []
##for (key,value) in feature_score:
##    fs.append("{0},{1}\n".format(key,value))
##
##with open('xgb0180716.csv','w') as f:
##        f.writelines("feature,score\n")
##        f.writelines(fs)
##














