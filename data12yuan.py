# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:52:18 2018

@author: 1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:18:53 2018


"""


import numpy as np
import pandas as pd
import datetime

data_train=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)
data_train.shape
#data_train=data_train.drop('is_trade',axis=1)
#data_train.shape
data_test=pd.read_csv('round1_ijcai_18_test_b_20180418.txt',delimiter=' ',header=0)
data_test.shape

#########删除训练数据的重复项
data_train.drop_duplicates(inplace=True)
#####给测试数据加上is_trade
#string2=data_train.columns.values.tolist()
#print(string2)
#string3=data_test.columns.values.tolist()
#print(string3)

data_test['is_trade']=10


def data_origin(data):
    
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
    #del data['context_timestamp']
    del data['context_id']
    
    ####查询词预测类目属性表数值化
    c1=data['predict_category_property'].apply(lambda x:x.split(';'))
    c11=c1.apply(lambda x:x[0].split(':')[0])
    data['predict_category_property_m1']=c1.apply(lambda x:x[0].split(':')[0])
    del data['predict_category_property']
    del data['instance_id']
    data.item_category_list=data.item_category_list.apply(lambda x:int(x))
    data.item_property_list_m1=data.item_property_list_m1.apply(lambda x:int(x))
    data.item_property_list_m2=data.item_property_list_m2.apply(lambda x:int(x))
    data.item_property_list_m3=data.item_property_list_m3.apply(lambda x:int(x))
    data.predict_category_property_m1=data.predict_category_property_m1.apply(lambda x:int(x))
    
    return data


#data_train1=data_origin(data_train.copy())
#data_test1=data_origin(data_test.copy())
#
#label_uid=data_test1['user_id'].unique()
#train_uid=data_train1[data_train1['user_id'].isin(label_uid)]
#

###########合并数据
data_train_test=pd.concat([data_train,data_test],ignore_index=True)
data1=data_origin(data_train_test)


#data=data1[data1.day==20]
data_y_train=data1[['day','is_trade']]

#data=data1.drop(['is_trade'],axis=1)


data=data1.copy()
###############合并训练数据和预测数据

#data=pd.concat([data_train_or,data_test_or],ignore_index=True)
data.isnull()

    
#--------------------------------item_id特征----------------------------
###商品浏览总次数
u1=data[['item_id']]
u1['item_is_see']=1
u1=u1.groupby(['item_id']).agg('sum').reset_index()
item_feature=u1

#####item_category_lis商品类的浏览次数
u1=data[['item_category_list']]
u1['item_category_see']=1
u1=u1.groupby(['item_category_list']).agg('sum').reset_index()
item_category_feature=u1
#######item_property_list_m1是商品属性浏览次数
u1=data[['item_property_list_m1']]
u1['item_property_list_m1_see']=1
u1=u1.groupby(['item_property_list_m1']).agg('sum').reset_index()
item_property_list_m1_feature=u1

######item_property_list_m2是商品属性浏览次数
u1=data[['item_property_list_m2']]
u1['item_property_list_m2_see']=1
u1=u1.groupby(['item_property_list_m2']).agg('sum').reset_index()
item_property_list_m2_feature=u1

######item_property_list_m3是商品属性浏览次数
u1=data[['item_property_list_m3']]
u1['item_property_list_m3_see']=1
u1=u1.groupby(['item_property_list_m3']).agg('sum').reset_index()
item_property_list_m3_feature=u1
#####item_brand商品不同品牌浏览总数
u1=data[['item_brand_id']]
u1['item_brand_see']=1
u1=u1.groupby(['item_brand_id']).agg('sum').reset_index()
item_brand_feature=u1
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

#all_item_data=data[['item_id','item_category_list','item_property_list_m1','item_property_list_m2','item_property_list_m3',
#                    'item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']]

all_item_data=data
all_item_data=pd.merge(all_item_data,item_feature,on=['item_id'],how='left')
all_item_data=pd.merge(all_item_data,item_category_feature,on=['item_category_list'],how='left')
all_item_data=pd.merge(all_item_data,item_property_list_m1_feature,on=['item_property_list_m1'],how='left')
all_item_data=pd.merge(all_item_data,item_property_list_m2_feature,on=['item_property_list_m2'],how='left')
all_item_data=pd.merge(all_item_data,item_property_list_m3_feature,on=['item_property_list_m3'],how='left')
all_item_data=pd.merge(all_item_data,item_brand_feature,on=['item_brand_id'],how='left')
all_item_data=pd.merge(all_item_data,item_item_category_see,on=['item_id','item_category_list'],how='left')
all_item_data=pd.merge(all_item_data,category_pro_m1_see,on=['item_category_list','item_property_list_m1'],how='left')
all_item_data=pd.merge(all_item_data,category_pro_m2_see,on=['item_category_list','item_property_list_m2'],how='left')
all_item_data=pd.merge(all_item_data,category_pro_m3_see,on=['item_category_list','item_property_list_m3'],how='left')
all_item_data=pd.merge(all_item_data,item_item_brand_see,on=['item_id','item_brand_id'],how='left')
all_item_data=pd.merge(all_item_data,brand_categoty_see,on=['item_brand_id','item_category_list'],how='left')
all_item_data=pd.merge(all_item_data,brand_pro_m1_see,on=['item_brand_id','item_property_list_m1'],how='left')
all_item_data=pd.merge(all_item_data,brand_pro_m2_see,on=['item_brand_id','item_property_list_m2'],how='left')
all_item_data=pd.merge(all_item_data,brand_pro_m3_see,on=['item_brand_id','item_property_list_m3'],how='left')

all_item_data.fillna(0)
#all_item_data.to_csv('all_item_data.csv',index=None)

##############------------------用户相关特征----------------
#####y用户浏览总次数
u1=data[['user_id']]
u1['user_id_see']=1
u1=u1.groupby('user_id').agg('sum').reset_index()
user_feature=u1

#####用户姓性别浏览次数
u1=data[['user_gender_id']]
u1['user_gender_see']=1
u1=u1.groupby('user_gender_id').agg('sum').reset_index()
user_gender_feature=u1

######不同年龄的浏览次数
u1=data[['user_age_level']]
u1['user_age_see']=1
u1=u1.groupby('user_age_level').agg('sum').reset_index()
user_age_feature=u1

########不同职业的浏览次数
u1=data[['user_occupation_id']]
u1['user_occu_see']=1
u1=u1.groupby('user_occupation_id').agg('sum').reset_index()
user_occu_feature=u1

######不同星机的浏览
u1=data[['user_star_level']]
u1['user_star_see']=1
u1=u1.groupby('user_star_level').agg('sum').reset_index()
user_star_feature=u1

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

#all_user_data=data[['user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']]
all_user_data=data

all_user_data=pd.merge(all_user_data,user_feature,on=['user_id'],how='left')
all_user_data=pd.merge(all_user_data,user_gender_feature,on=['user_gender_id'],how='left')
all_user_data=pd.merge(all_user_data,user_age_feature,on=['user_age_level'],how='left')
all_user_data=pd.merge(all_user_data,user_occu_feature,on=['user_occupation_id'],how='left')
all_user_data=pd.merge(all_user_data,user_star_feature,on=['user_star_level'],how='left')
all_user_data=pd.merge(all_user_data,user_gender_age,on=['user_gender_id','user_age_level'],how='left')

all_user_data=pd.merge(all_user_data,user_gender_ocu,on=['user_gender_id','user_occupation_id'],how='left')
all_user_data=pd.merge(all_user_data,user_age_ocu,on=['user_age_level','user_occupation_id'],how='left')

all_user_data.fillna(0)
#all_user_data.to_csv('all_user_data.csv',index=None)

#########----------------上下文特征--------------
####上下文page对应的浏览数和点击
u1=data[['context_page_id']]
u1['page_see']=1
u1=u1.groupby(['context_page_id']).agg('sum').reset_index()
page_feature=u1

######pre_cate预测浏览
u1=data[['predict_category_property_m1']]
u1['pre_cate_m1']=1
u1=u1.groupby(['predict_category_property_m1']).agg('sum').reset_index()
pre_cate_m1=u1

######page and cate
u1=data[['context_page_id','predict_category_property_m1']]
u1['page_cate_m1']=1
u1=u1.groupby(['context_page_id','predict_category_property_m1']).agg('sum').reset_index()
page_cate_m1=u1

#all_page_data=data[['context_timestamp','context_page_id','predict_category_property_m1']]
all_page_data=data

all_page_data=pd.merge(all_page_data,page_feature,on=['context_page_id'],how='left')
all_page_data=pd.merge(all_page_data,pre_cate_m1,on=['predict_category_property_m1'],how='left')
all_page_data=pd.merge(all_page_data,page_cate_m1,on=['context_page_id','predict_category_property_m1'],how='left')

all_page_data.fillna(0)
#all_page_data.to_csv('all_page_data.csv',index=None)

############--------------------店铺特征------------------
######店铺的浏览次数
u1=data[['shop_id']]
u1['shop_id_see']=1
u1=u1.groupby('shop_id').agg('sum').reset_index()
shop_feature=u1

#####店铺评价
u1=data[['shop_review_num_level']]
u1['review_see']=1
u1=u1.groupby(['shop_review_num_level']).agg('sum').reset_index()
shop_review_see=u1

####strashop
u1=data[['shop_star_level']]
u1['shop_star']=1
u1=u1.groupby('shop_star_level').agg('sum').reset_index()
shop_star=u1

all_shop_data=data[['shop_id','shop_review_num_level','shop_star_level']]
all_shop_data=data

all_shop_data=pd.merge(all_shop_data,shop_feature,on=['shop_id'],how='left')
all_shop_data=pd.merge(all_shop_data,shop_review_see,on=['shop_review_num_level'],how='left')
all_shop_data=pd.merge(all_shop_data,shop_star,on=['shop_star_level'],how='left')

all_shop_data.fillna(0)
#all_shop_data.to_csv('all_shop_data.csv',index=None)

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
all_two_data=pd.merge(all_two_data,user_item,on=['user_id','item_id'],how='left')
all_two_data=pd.merge(all_two_data,user_item_cate,on=['user_id','item_category_list'],how='left')
all_two_data=pd.merge(all_two_data,user_item_pro_m1,on=['user_id','item_property_list_m1'],how='left')
all_two_data=pd.merge(all_two_data,user_item_pro_m2,on=['user_id','item_property_list_m2'],how='left')
all_two_data=pd.merge(all_two_data,user_item_pro_m3,on=['user_id','item_property_list_m3'],how='left')
all_two_data=pd.merge(all_two_data,user_item_brand,on=['user_id','item_brand_id'],how='left')
all_two_data=pd.merge(all_two_data,user_hour,on=['user_id','hour'],how='left')
all_two_data=pd.merge(all_two_data,user_pre_cate_m1,on=['user_id','predict_category_property_m1'],how='left')
all_two_data=pd.merge(all_two_data,user_page,on=['user_id','context_page_id'],how='left')
#all_two_data=pd.merge(all_two_data,user_day_sum,on=['user_id','day'],how='left')
#all_two_data=pd.merge(all_two_data,user_day_cumsum,on=['user_id','day'],how='left')
all_two_data=pd.merge(all_two_data,user_shop,on=['user_id','shop_id'],how='left')

all_two_data=pd.merge(all_two_data,item_shop,on=['item_id','shop_id'],how='left')
all_two_data=pd.merge(all_two_data,item_cate_shop,on=['item_category_list','shop_id'],how='left')
all_two_data=pd.merge(all_two_data,item_pro_m1,on=['item_property_list_m1','shop_id'],how='left')
all_two_data=pd.merge(all_two_data,item_pro_m2,on=['item_property_list_m2','shop_id'],how='left')
all_two_data=pd.merge(all_two_data,item_pro_m3,on=['item_property_list_m3','shop_id'],how='left')
all_two_data=pd.merge(all_two_data,item_brand_shop,on=['item_brand_id','shop_id'],how='left')
all_two_data=pd.merge(all_two_data,item_page,on=['item_id','context_page_id'],how='left')
all_two_data=pd.merge(all_two_data,item_pre_cate,on=['item_id','predict_category_property_m1'],how='left')
all_two_data=pd.merge(all_two_data,item_cate_proper,on=['item_category_list','predict_category_property_m1'],how='left')
all_two_data=pd.merge(all_two_data,item_hour,on=['item_id','hour'],how='left')
all_two_data=pd.merge(all_two_data,shop_hour,on=['shop_id','hour'],how='left')
all_two_data=pd.merge(all_two_data,shop_pre_cate,on=['shop_id','predict_category_property_m1'],how='left')

all_two_data.fillna(0)
#all_two_data.to_csv('all_two_data.csv',index=None)

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

##----------------------用户和商品以及商户的此前出现次数，当天出现的次数，当天出现的次数累加等-----------------------

#####用户和day：在一天内的用户的累加浏览次数
u1=data[['user_id','day']]
u1['user_day']=1
user_day_sum=u1.groupby(['user_id','day']).agg('sum').reset_index()

u1['user_cumsum']=u1.groupby(['user_id'])['user_day'].cumsum()##############用户的累加浏览次数
u1['user_day_cumsum']=u1.groupby(['user_id','day'])['user_day'].cumsum()##############用户每天累加次数
user_day_cumsum=u1.drop(['user_id','day','user_day'],axis=1)



all_11=data[['user_id','day']]
all_11=pd.concat([all_11,user_day_cumsum],axis=1,join='outer')

#all_11=pd.merge(all_11,user_day_cumsum,on=['user_id','day'],how='left')
#ss=[]
#for i in range(300):
#    if all_11.user_id[i] !=user_day_cumsum.user_id[i]:
#        ss.append(i)

#####item和day：在一天内的商品编号的累加浏览次数
u1=data[['item_id','day']]
u1['item_day_times']=1
item_day_sum=u1.groupby(['item_id','day']).agg('sum').reset_index()

u1['item_cumsum']=u1.groupby(['item_id'])['item_day_times'].cumsum()##############商品id的累加浏览次数
u1['item_day_cumsum']=u1.groupby(['item_id','day'])['item_day_times'].cumsum()##############商品每天累加次数
item_day_cumsum=u1.drop(['item_id','day','item_day_times'],axis=1)



#####shop和day：在一天内的商店的编号的累加浏览次数
u1=data[['shop_id','day']]
u1['shop_day_times']=1
shop_day_sum=u1.groupby(['shop_id','day']).agg('sum').reset_index()

u1['shop_cumsum']=u1.groupby(['shop_id'])['shop_day_times'].cumsum()##############商品id的累加浏览次数
u1['shop_day_cumsum']=u1.groupby(['shop_id','day'])['shop_day_times'].cumsum()##############商品每天累加次数
shop_day_cumsum=u1.drop(['shop_id','day','shop_day_times'],axis=1)


########uese and itemm在倚天的累加浏览次数
u1=data[['user_id','item_id','day']]
u1['user_item_day_times']=1
user_item_day_sum=u1.groupby(['user_id','item_id','day']).agg('sum').reset_index()

u1['user_item_cumsum']=u1.groupby(['user_id','item_id'])['user_item_day_times'].cumsum()
u1['user_item_day_cumsum']=u1.groupby(['user_id','item_id','day'])['user_item_day_times'].cumsum()

user_item_day_cumsum=u1.drop(['user_id','item_id','day','user_item_day_times'],axis=1)

#############user  and  shopid的累加浏览次数
u1=data[['user_id','shop_id','day']]
u1['user_shop_day_times']=1
user_shop_day_sum=u1.groupby(['user_id','shop_id','day']).agg('sum').reset_index()

u1['user_shop_cumsum']=u1.groupby(['user_id','shop_id'])['user_shop_day_times'].cumsum()
u1['user_shop_day_cumsum']=u1.groupby(['user_id','shop_id','day'])['user_shop_day_times'].cumsum()
del u1['user_shop_day_times']
u1=pd.merge(u1,user_shop_day_sum,on=['user_id','shop_id','day'],how='left')
u1['is_last_user_shop']=0
u1.loc[(u1.user_shop_day_cumsum==u1.user_shop_day_times)&(u1.user_shop_day_times>1),'is_last_user_shop']=1

user_shop_day_cumsum=u1.drop(['user_id','shop_id','day'],axis=1)
#del u1['user_shop_day_times']
#user_shop_day_cumsum=u1


##########item  and  shop的日累加及星期累加次数
u1=data[['item_id','shop_id','day']]
u1['item_shop_day_times']=1
item_shop_day_sum=u1.groupby(['item_id','shop_id','day']).agg('sum').reset_index()

u1['item_shop_cumsum']=u1.groupby(['item_id','shop_id'])['item_shop_day_times'].cumsum()
u1['item_shop_day_cumsum']=u1.groupby(['item_id','shop_id','day'])['item_shop_day_times'].cumsum()
item_shop_day_cumsum=u1.drop(['item_id','shop_id','day','item_shop_day_times'],axis=1)

#del u1['item_shop_day_times']
#item_shop_day_cumsum=u1
##########uesr and category
u1=data[['user_id','item_category_list','day']]
u1['user_category_day_times']=1
user_categoyy_day_sum=u1.groupby(['user_id','item_category_list','day']).agg('sum').reset_index()

u1['user_category_cumsum']=u1.groupby(['user_id','item_category_list'])['user_category_day_times'].cumsum()
u1['user_category_day_cumsum']=u1.groupby(['user_id','item_category_list','day'])['user_category_day_times'].cumsum()
del u1['user_category_day_times']
u1=pd.merge(u1,user_categoyy_day_sum,on=['user_id','item_category_list','day'],how='left')
u1['is_last_user_category']=0
u1.loc[(u1.user_category_day_cumsum==u1.user_category_day_times)&(u1.user_category_day_times>1),'is_last_user_category']=1

user_category_day_cumsum=u1.drop(['user_id','item_category_list','day'],axis=1)
#user_category_day_cumsum=u1


##########item-categoty_day
u1=data[['item_category_list','day']]
u1['item_category_day_times']=1
item_category_day_sum=u1.groupby(['item_category_list','day']).agg('sum').reset_index()

u1['item_category_cumsum']=u1.groupby(['item_category_list'])['item_category_day_times'].cumsum()
u1['item_category_day_cumsum']=u1.groupby(['item_category_list','day'])['item_category_day_times'].cumsum()
#del u1['item_category_day_times']
#u1=pd.merge(u1,item_category_day_sum,on='item_category_list','day',how='left')
item_category_day_cumsum=u1.drop(['item_category_list','day','item_category_day_times'],axis=1)
#item_category_day_cumsum=u1

#aa=pd.merge(user_category_day_cumsum,item_category_day_cumsum,on=['item_category_list','day'],how='left')

#----------------------------------##################-------------------------------





all_time_data=data

all_time_data=pd.merge(all_time_data,time_delta_user,on=['user_id','context_timestamp'],how='left')
all_time_data=pd.merge(all_time_data,time_delta_item,on=['item_id','context_timestamp'],how='left')
all_time_data=pd.merge(all_time_data,time_delta_item_brand,on=['item_brand_id','context_timestamp'],how='left')
all_time_data=pd.merge(all_time_data,time_delta_shop,on=['shop_id','context_timestamp'],how='left')
all_time_data=pd.merge(all_time_data,time_delta_user_item,on=['user_id','item_id','context_timestamp'],how='left')
all_time_data=pd.merge(all_time_data,time_delta_user_brand,on=['user_id','item_brand_id','context_timestamp'],how='left')
all_time_data=pd.merge(all_time_data,time_delta_user_shop,on=['user_id','shop_id','context_timestamp'],how='left')

#all_time_data=pd.merge(all_time_data,time_delta_user_cate,on=['user_id','item_category_list','context_timestamp'],how='left')


all_time_data=pd.merge(all_time_data,user_day_sum,on=['user_id','day'],how='left')
all_time_data=pd.concat([all_time_data,user_day_cumsum],axis=1,join='outer')
all_time_data=pd.merge(all_time_data,item_day_sum,on=['item_id','day'],how='left')
all_time_data=pd.concat([all_time_data,item_day_cumsum],axis=1,join='outer')

all_time_data=pd.merge(all_time_data,item_category_day_sum,on=['item_category_list','day'],how='left')
all_time_data=pd.concat([all_time_data,item_category_day_cumsum],axis=1,join='outer')

all_time_data=pd.merge(all_time_data,shop_day_sum,on=['shop_id','day'],how='left')
all_time_data=pd.concat([all_time_data,shop_day_cumsum],axis=1,join='outer')
all_time_data=pd.merge(all_time_data,user_item_day_sum,on=['user_id','item_id','day'],how='left')
all_time_data=pd.concat([all_time_data,user_item_day_cumsum],axis=1,join='outer')

#all_time_data=pd.merge(all_time_data,user_shop_day_sum,on=['user_id','shop_id','day'],how='left')
all_time_data=pd.concat([all_time_data,user_shop_day_cumsum],axis=1,join='outer')

all_time_data=pd.merge(all_time_data,item_shop_day_sum,on=['item_id','shop_id','day'],how='left')
all_time_data=pd.concat([all_time_data,item_shop_day_cumsum],axis=1,join='outer')

#all_time_data=pd.merge(all_time_data,user_categoyy_day_sum,on=['user_id','item_category_list','day'],how='left')
all_time_data=pd.concat([all_time_data,user_category_day_cumsum],axis=1,join='outer')

strigng=all_time_data.columns.values.tolist()

all_time_data['is_cate_leak']=0
all_time_data.loc[(all_time_data.user_category_day_cumsum>1)&(all_time_data.user_category_day_cumsum==all_time_data.item_category_day_times),'is_cate_leak']=1

all_time_data['is_shop_leak']=0

all_time_data.loc[(all_time_data.user_shop_day_cumsum>1)&(all_time_data.user_shop_day_cumsum==all_time_data.shop_day_times),'is_shop_leak']=1



all_time_data.fillna(0)
#all_time_data.to_csv('all_time_data.csv',index=None)

    #############3----------------------拼接特征----------------------
string1=data.columns.values.tolist()
print(string1)

#all_item_data=pd.read_csv('all_item_data.csv')
#all_user_data=pd.read_csv('all_user_data.csv')
#all_page_data=pd.read_csv('all_page_data.csv')
#all_shop_data=pd.read_csv('all_shop_data.csv')
#all_two_data=pd.read_csv('all_two_data.csv')
#all_time_data=pd.read_csv('all_time_data.csv')

new_data=data
new_data=pd.merge(new_data,all_item_data,on=string1,how='left')
new_data=pd.merge(new_data,all_user_data,on=string1,how='left')
new_data=pd.merge(new_data,all_page_data,on=string1,how='left')
new_data=pd.merge(new_data,all_shop_data,on=string1,how='left')
new_data=pd.merge(new_data,all_two_data,on=string1,how='left')
new_data=pd.merge(new_data,all_time_data,on=string1,how='left')

#return new_data



new_train_data=new_data.copy()

#del new_train_data['user_brand_shop_see']
#del new_train_data['user_shop_see']
#del new_train_data['user_item_brand_see']
#del new_train_data['user_page_see']
del new_train_data['context_timestamp']

new_train_data.to_csv('new_train_data.csv',index=False)

string2=new_train_data.columns.values.tolist()
print(string2)

train=new_train_data[(new_train_data.day<24)].drop(['day','is_trade'],axis=1,inplace=False)
test=new_train_data[(new_train_data.day==24)].drop(['day','is_trade'],axis=1,inplace=False)

y_train=new_train_data[(new_train_data.day<24)][['is_trade']]
y_test=new_train_data[(new_train_data.day==24)][['is_trade']]

#y_train=data_y_train[(data_y_train.day<24)][['is_trade']]
#y_test=data_y_train[(data_y_train.day==24)][['is_trade']]



##############---------------lightGBM转换数据格式--------------------
import json
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import log_loss


print("load data")
X_train=train
y_train =y_train.iloc[:,0].values
y_test =y_test.iloc[:,0].values
X_test = test
string2=train.columns.values.tolist()
print(string2)              
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id']

string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_age_level','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2','item_property_list_m3','predict_category_property_m1']
#

#string3=['user_age_level','user_occupation_id','hour']        

###########-------------LIGHTgbm-------------------
#lgb0 = lgb.LGBMClassifier(
#        objective='binary',
##        metric='binary_logloss',
#        num_leaves=5,
##        max_depth=7,
#        learning_rate=0.025,
#        seed=2018,
#        reg_alpha=5,
#        colsample_bytree=0.9,
##         min_child_samples=8,
#        subsample=0.75,
#        n_estimators=20000)
#lgb0 = lgb.LGBMClassifier(
#        objective='binary',
#        # metric='binary_error',
#        num_leaves=6,
##        max_depth=7,
#        learning_rate=0.025,
#        seed=2018,
#        reg_alpha=6,
#        colsample_bytree=0.75,
##         min_child_samples=8,
#        subsample=0.9,
#        n_estimators=20000)  #########0.08070参数
#

lgb0 = lgb.LGBMClassifier(
        objective='binary',
#        metric='binary_logloss',
        num_leaves=5,
#        max_depth=7,
        learning_rate=0.025,
        seed=2018,
        reg_alpha=10,
#        reg_lambda=5,
        colsample_bytree=0.8,
#         min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
lgb_model = lgb0.fit(X_train, y_train, eval_set=[(X_test,y_test)],feature_name=string2, categorical_feature=string3,eval_metric='binary_logloss',early_stopping_rounds=200)
best_iter = lgb_model.best_iteration


#lgb.plot_importance(lgb_model)
#predictors = [i for i in X_train.columns]
#feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
#print(feat_imp)
#print(feat_imp.shape)
#
#string_feat=feat_imp[feat_imp.values==0].index.tolist()
#
#print(string_feat)
#
#for i in range(len(string_feat)):
#    del X_train[string_feat[i]]
#    del X_test[string_feat[i]]


#######24号的误差-
pred1= lgb_model.predict_proba(test)[:, 1]
print(log_loss(y_test,pred1))     

#############进行18-24号数据的训练
train=new_train_data[(new_train_data.day<25)].drop(['day','is_trade'],axis=1,inplace=False)
test=new_train_data[(new_train_data.day==25)].drop(['day','is_trade'],axis=1,inplace=False)
y_train=new_train_data[(new_train_data.day<25)][['is_trade']]
y_test=new_train_data[(new_train_data.day==25)][['is_trade']]

print("load data")
X_train=train
y_train =y_train.iloc[:,0].values
y_test =y_test.iloc[:,0].values
X_test = test
string2=train.columns.values.tolist()
print(string2)              
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_age_level','user_occupation_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1',]
string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_age_level','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3', 'predict_category_property_m1']


lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=5,
#        max_depth=7,
        learning_rate=0.025,
        seed=2018,
        reg_alpha=10,
#        reg_lambda=5,
        colsample_bytree=0.8,
#         min_child_samples=8,
        subsample=0.9,
        n_estimators=best_iter)




lgb_model = lgb0.fit(X_train, y_train,feature_name=string2,categorical_feature=string3)
pred2= lgb_model.predict_proba(X_test)[:, 1]

bao_cun=data_test[['instance_id']]
bao_cun['predicted_score']=pred2
bao_cun.to_csv('b20184201_791326.txt',index=False,sep=' ')




#
#
#
#
#
#



###############---------------lightGBM--------------------
#import json
#import lightgbm as lgb
#from sklearn.metrics import roc_curve, auc, roc_auc_score
#from sklearn.metrics import log_loss
#
#string2=train.columns.values.tolist()
#print(string2)
#
#print("load data")
#X_train=train
##df_test=pd.read_csv(path+"regression.train",header=None,sep='\t')
#y_train =y_train.iloc[:,0].values
#y_test =y_test.iloc[:,0].values
##X_train =np.array(X_train)
#X_test = test
#
#
#string2=train.columns.values.tolist()
#print(string2)
##string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1',]
#              
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','shop_id', 'item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3','hour', 'predict_category_property_m1',]
#                        
## create dataset for lightgbm
#lgb_train = lgb.Dataset(X_train, y_train,feature_name=string2,categorical_feature=string3)
#lgb_eval = lgb.Dataset(X_test, y_test,reference=lgb_train)
## specify your configurations as a dict
#
#params = {
#    'task': 'train',
#    'boosting_type': 'gbdt',
#    'objective': 'binary',
#    'metric': {'logloss','auc'},
#    'num_leaves': 63,
#    'max_depth':7,
##    'is_unbalance':'true',
#    'learning_rate': 0.1
#}
#
#num_round = 3000
#bst=lgb.cv(params, lgb_train, num_round, nfold=5,feature_name=string2,categorical_feature=string3,early_stopping_rounds=10)
#   
#print('Start training...')
## train
#gbm = lgb.train(params,
#                lgb_train,
#                num_boost_round=len(bst['auc-mean']),
#                feature_name=string2,
#                categorical_feature=string3,
#                valid_sets=lgb_eval,
#                early_stopping_rounds=10)
#
#print('Save model...')
## save model to file
#gbm.save_model('model.txt')
#print('Start predicting...')
## predict
#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
## eval
#print(y_pred)
#print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
#
#
#from sklearn.metrics import log_loss
#print(log_loss(y_test,y_pred))
#
#lgb.plot_importance(gbm)
#fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
#import matplotlib.pyplot as plt
#
#plt.figure()
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')



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


#######LIGHTGBM
###############lightGBM进行预测
#string2=train.columns.values.tolist()
#print(string2)
#string3=['item_id','item_category_list','item_brand_id','user_id','user_gender_id','user_age_level','user_occupation_id','shop_id','item_property_list_m1', 'item_property_list_m2', 'item_property_list_m3', 'hour', 'predict_category_property_m1']
#
##
##features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
##                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
##                'user_age_level', 'user_star_level',
##                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level','context_page_id','item_city_id'
##                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
##                ]
#
#
#import lightgbm as lgb
#from sklearn.metrics import log_loss
#from sklearn.grid_search import GridSearchCV
#
#from catboost import CatBoostClassifier
#cat_features=[0,1,2,7,8,9,10,12,13,14,20,21,22,23,24]
#
#model=CatBoostClassifier(iterations=100,learning_rate=0.05,depth=6,loss_function='Logloss')
#model.fit(train,y_train,cat_features)
#
#preds_proba=model.predict_proba(test)[:,1]
#
#print(log_loss(y_test,preds_proba))
#
#
##turn_params=[{'objective':['binary'],'learning_rate':[0.01,0.03,0.05,0.1],'n_estimators':[60,80,100],'max_depth':[6,7,8]}]
##
##clf=GridSearchCV(lgb.LGBMClassifier(seed=7),turn_params,scoring='roc_auc')
##clf.fit(train,y_train)
##print('best params of lgb is:',clf.best_params_)
#
#clf=lgb.LGBMClassifier(num_leaves=30,max_depth=4,learning_rate=0.1,n_estimators=300)
#clf.fit(train,y_train,feature_name=string2,categorical_feature=string3)
#
#
#y_pre=clf.predict_proba(test)[:,1]
#
#print(log_loss(y_test,y_pre))
#
#
#
#
###############进行全部数据的训练
#train=new_train_data.drop(['is_trade'],axis=1,inplace=False)
#y_train=new_train_data[['is_trade']]
#
#string2=train.columns.values.tolist()
#print(string2)
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id']
#
#import lightgbm as lgb
#from sklearn.metrics import log_loss
#from sklearn.grid_search import GridSearchCV
#
#clf=lgb.LGBMClassifier(num_leaves=50,max_depth=8,learning_rate=0.1,n_estimators=180)
#clf.fit(train,y_train,feature_name=string2,categorical_feature=string3)
#
#
#data_test=pd.read_csv('round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)
#
#new_data_test=data_precoss(data_test)
#
#y_pre_score=clf.predict_proba(new_data_test)[:,1]
#
#bao_cun=data_test[['instance_id']]
#bao_cun['predicted_score']=y_pre_score
#
#bao_cun.to_csv('zgw.txt',index=False,sep=' ')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#######划分需要进行编码的特征数据
#
#dataset1=new_feature_data.loc[:,['item_id','item_category_list','item_brand_id','item_city_id','user_id',
#               'user_gender_id','user_occupation_id','context_timestamp','context_page_id','shop_id']]
#
#
#
##dataset2=new_feature_data.loc[:,['item_property_list','item_property_list','item_price_level','item_sales_level',
##                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
##                     'predict_category_property','shop_review_num_level',
##                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
##                     'shop_score_description']]
#
#dataset2=new_feature_data.drop(['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_timestamp','context_page_id','shop_id'],axis=1,inplace=False)
#
#label=data.loc[:,'is_trade']
#
#
#
#
################lightGBM
##
##new_feature_data=new_feature_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
#
##X=data.drop(['instance_id','context_id','is_trade'],axis=1,inplace=False)
#X=new_feature_data
#
#y=data[['is_trade']]
#
#
#from sklearn.model_selection import train_test_split
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#
#
#import json
#import lightgbm as lgb
#
#from sklearn.metrics import roc_curve, auc, roc_auc_score
#
#
#print("load data")
##df_train=pd.read_csv(path+"regression.train",header=None,sep='\t')
##df_test=pd.read_csv(path+"regression.train",header=None,sep='\t')
##y_train = df_train[0].values
##y_test = df_test[0].values
##X_train = df_train.drop(0, axis=1).values
##X_test = df_test.drop(0, axis=1).values
#                     
#df_train=X_train
##df_test=pd.read_csv(path+"regression.train",header=None,sep='\t')
#y_train =y_train.iloc[:,0].values
#y_test =y_test.iloc[:,0].values
#X_train =np.array(X_train)
#X_test = np.array(X_test)
#                     
#                     
## create dataset for lightgbm
#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
## specify your configurations as a dict
#
##
##
#params = {
#    'task': 'train',
#    'boosting_type': 'gbdt',
#    'objective': 'binary',
#    'metric': {'logloss', 'auc'},
#    'num_leaves': 50,
#    'learning_rate': 0.02,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 0
#}
#
#
#
#
#
##string2=['item_id','item_category_list','item_property_list','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_id','user_gender_id',
##         'user_age_level','user_occupation_id','user_star_level','context_timestamp','context_page_id','predict_category_property','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
#
#string2=new_feature_data.columns.values.tolist()
#print(string2)
#string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_timestamp','context_page_id','shop_id']
#
#
#
#
#
#
#
#
#
#print('Start training...')
## train
#gbm = lgb.train(params,
#                lgb_train,
#                num_boost_round=3000,
#                feature_name=string2,
#                categorical_feature=string3,
#                valid_sets=lgb_eval,
#                early_stopping_rounds=10)
#
#
#
#print('Save model...')
## save model to file
#gbm.save_model('model.txt')
#print('Start predicting...')
## predict
#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
## eval
#print(y_pred)
#print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
#
#
#num_round = 300
#lgb.cv(params, lgb_train, num_round, nfold=5,feature_name=string2,categorical_feature=string3,early_stopping_rounds=10)
#
#
#
#
#fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
#import matplotlib.pyplot as plt
#
#plt.figure()
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
#
#
#from sklearn.metrics import log_loss
#print(log_loss(y_test,y_pred))
#
#
#
#
#import numpy as np
#logloss=np.zeros((len(y_test),1))
#p=y_pred
#import math as math
#for i in range(len(y_test)):
#    logloss[i]=y_test[i]*(math.log(10,p[i]))+(1-y_test[i])*(math.log(10,1-p[i]))
#    
#logloss=-1/len(y_test)*(np.sum(logloss))
#print('losloss is that',logloss)
#
#
#
#
#
#
#
#
#
#
#
#
#
###########进行独热编码
#t1=dataset1.astype(str)
#data_t1=pd.get_dummies(t1)
############可以验证一下id类型的种类个数
####item:3695   item_brand_id:1101   item_city_id:99  user_id:13573   user_gender_id:4
#####user_occupation_id:5  context_timestamp:5    shop_id:2015   item_category_list:13
#t=pd.DataFrame(dataset2.loc[:,'item_category_list'])
#
#t['instance_id_count']=1
#t=t.groupby('item_category_list').agg('sum').reset_index()
##########
#
#
#
############将数值型特征进行归一化
#dataset2_ave=dataset2.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
#
#
#######将id类特征进行归一化
#
#dataset1_ave=dataset1.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
#
#
#######输入GBDT进行特征融合
#data_new=data.drop(['context_id','is_trade'],axis=1,inplace=False)
#data_new=data_new.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
#
#
#
#X=data_new
#y=label
##X=np.array(X)
##y=np.array(y)
#
#from sklearn.model_selection import train_test_split
#
#from sklearn.metrics import roc_curve, auc, roc_auc_score
#
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
#                              GradientBoostingClassifier)
#
#from sklearn.preprocessing import OneHotEncoder
#
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
##X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
##                                                            y_train,
##                                                            test_size=0.5)
#
#
#
#
##X_train_11=X_train[['item_id','item_category_list','item_brand_id','item_city_id','user_id',
##               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]
##               
##X_train_lr11=X_train[['item_id','item_category_list','item_brand_id','item_city_id','user_id',
##               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]    
#
#
#
#grd_11 = GradientBoostingClassifier()
#grd_enc_11 = OneHotEncoder()
#grd_lm_11 = LogisticRegression()
#grd_11.fit(X_train, y_train)
#
#grd_enc_11.fit(grd_11.apply(X_train)[:, :, 0])
#
#lr11=grd_enc_11.transform(grd_11.apply(X_train)[:, :, 0])
#
#
#
#grd_lm_11.fit(lr11, y_train)
#
#
#ltest11=grd_enc_11.transform(grd_11.apply(X_test)[:, :, 0])
#
#
#y_pred_grd_lm = grd_lm_11.predict_proba(ltest11)[:, 1]
#    
#p=y_pred_grd_lm#############预测的概率
#
#fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
#
#xgb_lr_auc = roc_auc_score(y_test, y_pred_grd_lm)
#print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc)
#
#import matplotlib.pyplot as plt
#
#plt.figure()
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
#
#logloss=np.zeros((len(y_test),1))
#import math as math
#for i in range(len(y_test)):
#    logloss[i]=y_test[i]*(math.log(10,p[i]))+(1-y_test[i])*(math.log(10,1-p[i]))
#    
#logloss=-1/len(y_test)*(np.sum(logloss))
#print('losloss is that',logloss)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
######对id类和非id类分别建树
#
#
#X_train_1=X_train[['item_id']]
#X_train_2=X_train[['item_category_list']]
#X_train_3=X_train[['item_brand_id']]
#X_train_4=X_train[['item_city_id']]
#X_train_5=X_train[['user_id']]
#X_train_6=X_train[['user_gender_id']]
#X_train_7=X_train[['user_occupation_id']]
#X_train_8=X_train[['context_timestamp']]
#X_train_9=X_train[['shop_id']]
#X_train_10=X_train[['item_property_list','item_property_list','item_price_level','item_sales_level',
#                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
#                     'context_page_id','predict_category_property','shop_review_num_level',
#                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
#                     'shop_score_description']]
#                     
#
##X_train_lr1=X_train_lr[['item_id']]
##X_train_lr2=X_train_lr[['item_category_list']]
##X_train_lr3=X_train_lr[['item_brand_id']]
##X_train_lr4=X_train_lr[['item_city_id']]
##X_train_lr5=X_train_lr[['user_id']]
##X_train_lr6=X_train_lr[['user_gender_id']]
##X_train_lr7=X_train_lr[['user_occupation_id']]
##X_train_lr8=X_train_lr[['context_timestamp']]
##X_train_lr9=X_train_lr[['shop_id']]
##X_train_lr10=X_train_lr[['item_property_list','item_property_list','item_price_level','item_sales_level',
##                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
##                     'context_page_id','predict_category_property','shop_review_num_level',
##                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
##                     'shop_score_description']]
#
#n_estimator =1
#
#grd_1 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_1 = OneHotEncoder()
#grd_lm_1 = LogisticRegression()
#grd_1.fit(X_train_1, y_train)
#
#grd_enc_1.fit(grd_1.apply(X_train_1)[:, :, 0])
#
#lr1=grd_enc_1.transform(grd_1.apply(X_train_1)[:, :, 0])
#
#
#
#
#grd_2 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_2 = OneHotEncoder()
#grd_lm_2 = LogisticRegression()
#grd_2.fit(X_train_2, y_train)
#
#grd_enc_2.fit(grd_2.apply(X_train_2)[:, :, 0])
#
#lr2=grd_enc_2.transform(grd_2.apply(X_train_2)[:, :, 0])
#
#
#grd_3 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_3 = OneHotEncoder()
#grd_lm_3 = LogisticRegression()
#grd_3.fit(X_train_3, y_train)
#
#grd_enc_3.fit(grd_3.apply(X_train_3)[:, :, 0])
#
#lr3=grd_enc_3.transform(grd_3.apply(X_train_3)[:, :, 0])
#
#
#grd_4 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_4 = OneHotEncoder()
#grd_lm_4 = LogisticRegression()
#grd_4.fit(X_train_4, y_train)
#
#grd_enc_4.fit(grd_4.apply(X_train_4)[:, :, 0])
#
#lr4=grd_enc_4.transform(grd_4.apply(X_train_4)[:, :, 0])
#
#
#
#grd_5 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_5 = OneHotEncoder()
#grd_lm_5 = LogisticRegression()
#grd_5.fit(X_train_5, y_train)
#
#grd_enc_5.fit(grd_5.apply(X_train_5)[:, :, 0])
#
#lr5=grd_enc_5.transform(grd_5.apply(X_train_5)[:, :, 0])
#
#
#grd_6 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_6 = OneHotEncoder()
#grd_lm_6 = LogisticRegression()
#grd_6.fit(X_train_6, y_train)
#
#grd_enc_6.fit(grd_6.apply(X_train_6)[:, :, 0])
#
#lr6=grd_enc_6.transform(grd_6.apply(X_train_6)[:, :, 0])
#
#
#
#grd_7 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_7 = OneHotEncoder()
#grd_lm_7 = LogisticRegression()
#grd_7.fit(X_train_7, y_train)
#
#grd_enc_7.fit(grd_7.apply(X_train_7)[:, :, 0])
#
#lr7=grd_enc_7.transform(grd_7.apply(X_train_7)[:, :, 0])
#
#
#
#grd_8 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_8 = OneHotEncoder()
#grd_lm_8 = LogisticRegression()
#grd_8.fit(X_train_8, y_train)
#
#grd_enc_8.fit(grd_8.apply(X_train_8)[:, :, 0])
#
#lr8=grd_enc_8.transform(grd_8.apply(X_train_8)[:, :, 0])
#
#
#grd_9 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_9 = OneHotEncoder()
#grd_lm_9 = LogisticRegression()
#grd_9.fit(X_train_9, y_train)
#
#grd_enc_9.fit(grd_9.apply(X_train_9)[:, :, 0])
#
#lr9=grd_enc_9.transform(grd_9.apply(X_train_9)[:, :, 0])
#
#
#
#grd_10 = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc_10 = OneHotEncoder()
#grd_lm_10 = LogisticRegression()
#grd_10.fit(X_train_10, y_train)
#
#grd_enc_10.fit(grd_10.apply(X_train_10)[:, :, 0])
#
#lr10=grd_enc_10.transform(grd_10.apply(X_train_10)[:, :, 0])
#
#
#
#
#
#
#
#
#lr1=lr1.toarray()
#lr2=lr2.toarray()
#lr3=lr3.toarray()
#lr4=lr4.toarray()
#lr5=lr5.toarray()
#lr6=lr6.toarray()
#lr7=lr7.toarray()
#lr8=lr8.toarray()
#lr9=lr9.toarray()
#lr10=lr10.toarray()
#
#
#
#result=np.concatenate([lr1,lr2,lr3,lr4,lr5,lr6,lr7,lr8,lr9,lr10],axis=1)
#
#y_train=np.array(y_train)
#
########result为总的GBDT提取出来的特征向量
#grd_lm=LogisticRegression()
#
#
#grd_lm.fit(result, y_train)
#
#
#
#
###############测试集做同样处理
#X_test_1=X_test[['item_id']]
#X_test_2=X_test[['item_category_list']]
#X_test_3=X_test[['item_brand_id']]
#X_test_4=X_test[['item_city_id']]
#X_test_5=X_test[['user_id']]
#X_test_6=X_test[['user_gender_id']]
#X_test_7=X_test[['user_occupation_id']]
#X_test_8=X_test[['context_timestamp']]
#X_test_9=X_test[['shop_id']]
#X_test_10=X_test[['item_property_list','item_property_list','item_price_level','item_sales_level',
#                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
#                     'context_page_id','predict_category_property','shop_review_num_level',
#                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
#                     'shop_score_description']]
#
#
##
##grd_enc_9.fit(grd_9.apply(X_train_9)[:, :, 0])
##
##lr9=grd_enc_9.transform(grd_9.apply(X_train_lr9)[:, :, 0])
#
#
#ltest1=grd_enc_1.transform(grd_1.apply(X_test_1)[:, :, 0])
#ltest2=grd_enc_2.transform(grd_2.apply(X_test_2)[:, :, 0])
#ltest3=grd_enc_3.transform(grd_3.apply(X_test_3)[:, :, 0])
#ltest4=grd_enc_4.transform(grd_4.apply(X_test_4)[:, :, 0])
#ltest5=grd_enc_5.transform(grd_5.apply(X_test_5)[:, :, 0])
#ltest6=grd_enc_6.transform(grd_6.apply(X_test_6)[:, :, 0])
#ltest7=grd_enc_7.transform(grd_7.apply(X_test_7)[:, :, 0])
#ltest8=grd_enc_8.transform(grd_8.apply(X_test_8)[:, :, 0])
#ltest9=grd_enc_9.transform(grd_9.apply(X_test_9)[:, :, 0])
#ltest10=grd_enc_10.transform(grd_10.apply(X_test_10)[:, :, 0])
#
#ltest1=ltest1.toarray()
#ltest2=ltest2.toarray()
#ltest3=ltest3.toarray()
#ltest4=ltest4.toarray()
#ltest5=ltest5.toarray()
#ltest6=ltest6.toarray()
#ltest7=ltest7.toarray()
#ltest8=ltest8.toarray()
#ltest9=ltest9.toarray()
#ltest10=ltest10.toarray()
#
#
#new_test=np.concatenate([ltest1,ltest2,ltest3,ltest4,ltest5,ltest6,ltest7,ltest8,ltest9,ltest10],axis=1)
#
#
#y_test=np.array(y_test)
#
#
#y_pred_grd_lm = grd_lm.predict_proba(new_test)[:, 1]
#    
#p=y_pred_grd_lm#############预测的概率
#
#fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
#
#
#
#xgb_lr_auc = roc_auc_score(y_test, y_pred_grd_lm)
#print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc)
#
#
#
#import matplotlib.pyplot as plt
#
#plt.figure()
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
#
#logloss=np.zeros((len(y_test),1))
#
#import math as math
#for i in range(len(y_test)):
#    logloss[i]=y_test[i]*(math.log(10,p[i]))+(1-y_test[i])*(math.log(10,1-p[i]))
#    
#logloss=-1/len(y_test)*(np.sum(logloss))
#print('losloss is that',logloss)
#
#
#
#
##
##
##x1=pd.DataFrame(dataset1.loc[:,['user_id', 'user_gender_id']])
##x1['count']=1
##x1=x1.groupby(['user_id','user_gender_id']).agg('sum').reset_index()
##
##
##t1=dataset1.loc[:,['item_id']]
##t1=t1.apply(lambda x:x.astype(str))
##t11=pd.get_dummies(t1)
##    
##
###t1=np.array([[1],[2],[3],[5]])
###
###from sklearn import preprocessing
###enc=preprocessing.OneHotEncoder()
###enc.fit(t1)
###aa=enc.transform(t1)
###print(aa)
##
##t2=t1.loc[:,'item_id'].apply(lambda x :x<0)
##
##from numpy import argmax
##from sklearn.preprocessing import LabelEncoder
##from sklearn.preprocessing import OneHotEncoder
##
##d1=np.array(data[['item_id']])
##print(d1)
##
##label_encoder=LabelEncoder()
##integer_encoded=label_encoder.fit_transform(d1)
##
##
#######广告商品的特征
##a13=t1
##t=a13
##t['instance_id_count']=1
##t=t.groupby('item_id').agg('sum').reset_index()
