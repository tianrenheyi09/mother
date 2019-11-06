# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:00:10 2018

@author: 1
"""



import pandas as pd

#data=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',nrows=10000,delimiter=' ',header=0)
data=pd.read_csv('D:/mother/round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)
data.shape

#aa=data[['is_trade']]
#
#aa.iloc[:,0].value_counts()

data.isnull().any()

a1=data[['item_category_list']]

import numpy as np
a11=pd.DataFrame(np.zeros((len(a1),1)))
a12=pd.DataFrame(np.zeros((len(a1),1)))
a1=a1['item_category_list'].apply(lambda x:x.split(';'))

a11=a1.apply(lambda x :x[0])
a12=a1.apply(lambda x :x[1])
a11=pd.DataFrame(a11)
a12=pd.DataFrame(a12)
a12=a12['item_category_list'].apply(lambda x:int(x))

####令商品的category——list为a12
data[['item_category_list']]=a12
######广告商品类目表数值化
    
    
    
    
######广告商品属性列表数值化
b1=data[['item_property_list']]
b1=b1['item_property_list'].apply(lambda x:x.split(';'))
b1=b1.apply(lambda x:len(x))

data[['item_property_list']]=b1


###########上下文信息
####查询词预测类目属性表数值化
c1=data[['predict_category_property']]

c1=c1['predict_category_property'].apply(lambda x:x.split(';'))
c1=c1.apply(lambda x:len(x))

data[['predict_category_property']]=c1

#########时间戳时间格式化
import datetime
c2=data[['context_timestamp']]

c2=c2['context_timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))

c3=c2.astype(str).apply(lambda x:x.split(' '))
c3=c3.apply(lambda x:x[1])

c31=c3.apply(lambda x:(int(x[0])*10+int(x[1]))%24)

c32=c3.apply(lambda x:(int(x[3])*10+int(x[4]))%60)


def time_duan(x,y):
    if (x<=7 and y<=59):
        return 1
    elif (x>=8 and x<11 and y<=59):
        return 2
    elif (x>=11 and x<13 and y<=59):
        return 3
    elif (x>=13 and x<18 and y<=59):
        return 4
    else :
        return 5
    
c33=np.zeros((len(c31),1))

for i in range(len(c31)):
    c33[i]=time_duan(c31[i],c32[i])

data[['context_timestamp']]=c33

#######根据instance_id进行重复项的删除
data=data.drop_duplicates(['instance_id'])

data.shape


data.dtypes
###############挖掘其他隐含特征
u=data[['item_id']]
u.drop_duplicates(inplace=True)

###商品浏览总次数
u1=data[['item_id']]
u1['item_is_see']=1
u1=u1.groupby(['item_id']).agg('sum').reset_index()

item_feature=pd.merge(u,u1,on=['item_id'],how='left')

#####商品成交总次数
u2=data[['item_id','is_trade']]
u2=u2[(u2.is_trade==1)][['item_id']]

u2['item_is_trade']=1
u2=u2.groupby(['item_id']).agg('sum').reset_index()


item_feature=pd.merge(item_feature,u2,on=['item_id'],how='left')

######商品成交率


item_feature=item_feature.fillna(0)
item_feature['item_%%trade']=item_feature.item_is_trade/item_feature.item_is_see


#####商品不同品牌浏览总数
u1=data[['item_brand_id']]
u1['item_brand_see']=1
u1=u1.groupby(['item_brand_id']).agg('sum').reset_index()

######商品不同品牌成交次数
u2=data[(data.is_trade==1)][['item_brand_id']]
u2['item_brand_trade']=1
u2=u2.groupby(['item_brand_id']).agg('sum').reset_index()

######s商品不同同品成交率
item_brand_feature=pd.merge(u1,u2,on=['item_brand_id'],how='left')
item_brand_feature=item_brand_feature.fillna(0)
item_brand_feature['item_brand_%%trade']=item_brand_feature.item_brand_trade/item_brand_feature.item_brand_see


#####y用户浏览总次数
u1=data[['user_id']]
u1['user_id_see']=1
u1=u1.groupby('user_id').agg('sum').reset_index()

####用户成交次数
u2=data[(data.is_trade==1)][['user_id']]
u2['user_trade']=1
u2=u2.groupby('user_id').agg('sum').reset_index()

#####用户历史成交率
user_feature=pd.merge(u1,u2,on=['user_id'],how='left')
user_feature=user_feature.fillna(0)
user_feature['user_%%trade']=user_feature.user_trade/user_feature.user_id_see

####上下文page对应的浏览数和点击

u1=data[['context_page_id']]
u1['page_see']=1
u1=u1.groupby(['context_page_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['context_page_id']]
u2['page_trade']=1
u2=u2.groupby(['context_page_id']).agg('sum').reset_index()


page_feature=pd.merge(u1,u2,on=['context_page_id'],how='left')
page_feature=page_feature.fillna(0)


page_feature['page_%%trade']=page_feature.page_trade/page_feature.page_see

######店铺的浏览次数
#u1=data[['context_timestamp']]
#u1['context_timestamp_see']=1
#u1=u1.groupby('context_timestamp').agg('sum').reset_index()

u1=data[['shop_id']]
u1['shop_id_see']=1
u1=u1.groupby('shop_id').agg('sum').reset_index()

####店铺的成交次数
u2=data[(data.is_trade==1)][['shop_id']]
u2['shop_id_trade']=1
u2=u2.groupby('shop_id').agg('sum').reset_index()
####店铺的成交率
shop_feature=pd.merge(u1,u2,on=['shop_id'],how='left')
shop_feature=shop_feature.fillna(0)
shop_feature['shop_%%trade']=shop_feature.shop_id_trade/shop_feature.shop_id_see

#####用户和商品编号的之间的特征
u1=data[['user_id','item_id']]
u1['user_item_see']=1
u1=u1.groupby(['user_id','item_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['user_id','item_id']]
u2['user_item_trade']=1
u2=u2.groupby(['user_id','item_id']).agg('sum').reset_index()

user_item_feature=pd.merge(u1,u2,on=['user_id','item_id'],how='left')
user_item_feature=user_item_feature.fillna(0)

#########用户和商品 品牌之间的特征
u1=data[['user_id','item_brand_id']]
u1['user_item_brand_see']=1
u1=u1.groupby(['user_id','item_brand_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['user_id','item_brand_id']]
u2['user_item_brand_trade']=1
u2=u2.groupby(['user_id','item_brand_id']).agg('sum').reset_index()

user_brand_feature=pd.merge(u1,u2,on=['user_id','item_brand_id'],how='left')
user_brand_feature=user_brand_feature.fillna(0)


#########用户和上下文时间的特征
u1=data[['user_id','context_timestamp']]
u1['user_time_see']=1
u1=u1.groupby(['user_id','context_timestamp']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['user_id','context_timestamp']]
u2['user_time_trade']=1
u2=u2.groupby(['user_id','context_timestamp']).agg('sum').reset_index()

user_time_feature=pd.merge(u1,u2,on=['user_id','context_timestamp'],how='left')
user_time_feature=user_time_feature.fillna(0)


###########用户和店铺的特征
u1=data[['user_id','shop_id']]
u1['user_shop_see']=1
u1=u1.groupby(['user_id','shop_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['user_id','shop_id']]
u2['user_shop_trade']=1
u2=u2.groupby(['user_id','shop_id']).agg('sum').reset_index()

user_shop_feature=pd.merge(u1,u2,on=['user_id','shop_id'],how='left')
user_shop_feature=user_shop_feature.fillna(0)


######用户和上下文page之间的联系
u1=data[['user_id','context_page_id']]
u1['user_page_see']=1
u1=u1.groupby(['user_id','context_page_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['user_id','context_page_id']]
u2['user_page_trade']=1
u2=u2.groupby(['user_id','context_page_id']).agg('sum').reset_index()

user_page_feature=pd.merge(u1,u2,on=['user_id','context_page_id'],how='left')
user_page_feature=user_page_feature.fillna(0)


#############用户和店铺以及品牌的特征
u1=data[['user_id','item_brand_id','shop_id']]
u1['user_brand_shop_see']=1
u1=u1.groupby(['user_id','item_brand_id','shop_id']).agg('sum').reset_index()

u2=data[data.is_trade==1][['user_id','item_brand_id','shop_id']]
u2['user_brand_shop_trade']=1

u2=u2.groupby(['user_id','item_brand_id','shop_id']).agg('sum').reset_index()

user_brand_shop_feature=pd.merge(u1,u2,on=['user_id','item_brand_id','shop_id'],how='left')

user_brand_shop_feature=user_brand_shop_feature.fillna(0)


######用户和商品品牌以及商品页码的特征
u1=data[['user_id','item_brand_id','context_page_id']]
u1['user_brand_page_see']=1
u1=u1.groupby(['user_id','item_brand_id','context_page_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['user_id','item_brand_id','context_page_id']]
u2['user_brand_page_trade']=1
u2=u2.groupby(['user_id','item_brand_id','context_page_id']).agg('sum').reset_index()

user_brand_page_feature=pd.merge(u1,u2,on=['user_id','item_brand_id','context_page_id'],how='left')
user_brand_page_feature=user_brand_page_feature.fillna(0)



########商品品牌编号和上下文广告商品展示编号的特征
u1=data[['item_brand_id','context_page_id']]
u1['brand_page_see']=1
u1=u1.groupby(['item_brand_id','context_page_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['item_brand_id','context_page_id']]
u2['brand_page_trade']=1
u2=u2.groupby(['item_brand_id','context_page_id']).agg('sum').reset_index()

brand_page_feature=pd.merge(u1,u2,on=['item_brand_id','context_page_id'],how='left')
brand_page_feature=brand_page_feature.fillna(0)

#######上下文展示编号和店铺id间的特征
u1=data[['context_page_id','shop_id']]
u1['page_shop_see']=1
u1=u1.groupby(['context_page_id','shop_id']).agg('sum').reset_index()

u2=data[(data.is_trade==1)][['context_page_id','shop_id']]
u2['page_shop_trade']=1
u2=u2.groupby(['context_page_id','shop_id']).agg('sum').reset_index()


page_shop_feature=pd.merge(u1,u2,on=['context_page_id','shop_id'],how='left')
page_shop_feature=page_shop_feature.fillna(0)





###########合并另外提取的特征
new_feature_data=data.drop(['instance_id','context_id','is_trade'],axis=1,inplace=False)

new_feature_data=pd.merge(new_feature_data,item_feature,on='item_id',how='left')
new_feature_data=pd.merge(new_feature_data,item_brand_feature,on='item_brand_id',how='left')
new_feature_data=pd.merge(new_feature_data,user_feature,on='user_id',how='left')
new_feature_data=pd.merge(new_feature_data,page_feature,on='context_page_id',how='left')
new_feature_data=pd.merge(new_feature_data,shop_feature,on='shop_id',how='left')
new_feature_data=pd.merge(new_feature_data,user_item_feature,on=['user_id','item_id'],how='left')


new_feature_data=pd.merge(new_feature_data,user_brand_feature,on=['user_id','item_brand_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_time_feature,on=['user_id','context_timestamp'],how='left')
new_feature_data=pd.merge(new_feature_data,user_shop_feature,on=['user_id','shop_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_page_feature,on=['user_id','context_page_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_brand_shop_feature,on=['user_id','item_brand_id','shop_id'],how='left')
new_feature_data=pd.merge(new_feature_data,user_brand_page_feature,on=['user_id','item_brand_id','context_page_id'],how='left')
new_feature_data=pd.merge(new_feature_data,brand_page_feature,on=['item_brand_id','context_page_id'],how='left')
new_feature_data=pd.merge(new_feature_data,page_shop_feature,on=['context_page_id','shop_id'],how='left')


new_feature_data.to_csv('new_feature_data.csv',index=None)

