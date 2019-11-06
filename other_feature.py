# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:51:11 2018

@author: 1
"""
import numpy as np
import pandas as pd
import datetime

data_final=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)
#data=pd.read_csv('D:/mother/round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)
data_final.shape

data1=data_final.copy()

#######根据instance_id进行重复项的删除
data1=data1.drop_duplicates(['instance_id'])  
data1.shape  
data1.dtypes


#########时间戳时间格式化
c2=data1[['context_timestamp']]
c2=c2['context_timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))
c3=c2.astype(str).apply(lambda x:x.split(' '))
data1['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
data1['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))

del data1['context_timestamp']


######截取数据
data=data1[(data1.day<24)]
data_24=data1[(data1.day==24)]


############
a1=data[['item_category_list']]
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

#
#c2=data[['context_timestamp']]
#c2=c2['context_timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))
#c3=c2.astype(str).apply(lambda x:x.split(' '))
#data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
#data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))
#
#del data['context_timestamp']



#data[['context_timestamp']]=c33

#######根据instance_id进行重复项的删除
data=data.drop_duplicates(['instance_id'])  
data.shape  
data.dtypes
###############挖掘其他隐含特征
u=data[['item_id']]
u.drop_duplicates(inplace=True)


####用户在同一时间段的浏览次数
u1=data[['user_id','hour']]
u1['user_hour_see']=1
u1=u1.groupby(['user_id','hour']).agg('sum').reset_index()

user_hour_see=u1

###用户浏览谋商品品牌的时间间隔
u1=data[['user_id','item_brand_id','hour']]
u1['user_brand_hour']=1
u1=u1.groupby(['user_id','item_brand_id','hour']).agg('sum').reset_index()

user_brand_hour_see=u1

data=pd.merge(data,user_hour_see,on=['user_id','hour'],how='left')
data=pd.merge(data,user_brand_hour_see,on=['user_id','item_brand_id','hour'],how='left')
data.fillna(0)


###商品浏览总次数



#####商品成交总次数
u2=data[['item_id','is_trade']]
u2=u2[(u2.is_trade==1)][['item_id']]

u2['item_is_trade']=1
u2=u2.groupby(['item_id']).agg('sum').reset_index()

item_is_trade=u2


#######商品不同品牌成交次数
u2=data[(data.is_trade==1)][['item_brand_id']]
u2['item_brand_trade']=1
u2=u2.groupby(['item_brand_id']).agg('sum').reset_index()

item_brand_trade=u2
#######s商品不同同品成交率


#####用户成交次数
u2=data[(data.is_trade==1)][['user_id']]
u2['user_trade']=1
u2=u2.groupby('user_id').agg('sum').reset_index()

user_trade=u2
#

####上下文page对应的浏览数和点击


u2=data[(data.is_trade==1)][['context_page_id']]
u2['page_trade']=1
u2=u2.groupby(['context_page_id']).agg('sum').reset_index()

page_trade=u2
#
#


#####店铺的成交次数
u2=data[(data.is_trade==1)][['shop_id']]
u2['shop_id_trade']=1
u2=u2.groupby('shop_id').agg('sum').reset_index()

shop_id_trade=u2


#####用户和商品编号的之间的特征

u2=data[(data.is_trade==1)][['user_id','item_id']]
u2['user_item_trade']=1
u2=u2.groupby(['user_id','item_id']).agg('sum').reset_index()

user_item_trade=u2



#########用户和商品 品牌之间的特征


u2=data[(data.is_trade==1)][['user_id','item_brand_id']]
u2['user_item_brand_trade']=1
u2=u2.groupby(['user_id','item_brand_id']).agg('sum').reset_index()

user_item_brand_trade=u2


###########用户和店铺的特征

u2=data[(data.is_trade==1)][['user_id','shop_id']]
u2['user_shop_trade']=1
u2=u2.groupby(['user_id','shop_id']).agg('sum').reset_index()

user_shop_trade=u2


######用户和上下文page之间的联系

u2=data[(data.is_trade==1)][['user_id','context_page_id']]
u2['user_page_trade']=1
u2=u2.groupby(['user_id','context_page_id']).agg('sum').reset_index()

user_page_trade=u2


#############用户和店铺以及品牌的特征


u2=data[data.is_trade==1][['user_id','item_brand_id','shop_id']]
u2['user_brand_shop_trade']=1

u2=u2.groupby(['user_id','item_brand_id','shop_id']).agg('sum').reset_index()

user_brand_shop_trade=u2
#
######用户和商品品牌以及商品页码的特征



u2=data[(data.is_trade==1)][['user_id','item_brand_id','context_page_id']]
u2['user_brand_page_trade']=1
u2=u2.groupby(['user_id','item_brand_id','context_page_id']).agg('sum').reset_index()

user_brand_page_trade=u2
########商品品牌编号和上下文广告商品展示编号的特征


u2=data[(data.is_trade==1)][['item_brand_id','context_page_id']]
u2['brand_page_trade']=1
u2=u2.groupby(['item_brand_id','context_page_id']).agg('sum').reset_index()

brand_page_trade=u2
#######上下文展示编号和店铺id间的特征


u2=data[(data.is_trade==1)][['context_page_id','shop_id']]
u2['page_shop_trade']=1
u2=u2.groupby(['context_page_id','shop_id']).agg('sum').reset_index()

page_shop_trade=u2





def data_process(data):
 
#    a1=data[['item_category_list']]
#    
#
#    a11=pd.DataFrame(np.zeros((len(a1),1)))
#    a12=pd.DataFrame(np.zeros((len(a1),1)))
#    a1=a1['item_category_list'].apply(lambda x:x.split(';'))
#    
#    a11=a1.apply(lambda x :x[0])
#    a12=a1.apply(lambda x :x[1])
#    a11=pd.DataFrame(a11)
#    a12=pd.DataFrame(a12)
#    a12=a12['item_category_list'].apply(lambda x:int(x))
#    
#    ####令商品的category——list为a12
#    data[['item_category_list']]=a12
#    ######广告商品类目表数值化
#            
#    ######广告商品属性列表数值化
#    b1=data[['item_property_list']]
#    b1=b1['item_property_list'].apply(lambda x:x.split(';'))
#    b1=b1.apply(lambda x:len(x))
#    
#    data[['item_property_list']]=b1
#    
#    
#    ###########上下文信息
#    ####查询词预测类目属性表数值化
#    c1=data[['predict_category_property']]
#    
#    c1=c1['predict_category_property'].apply(lambda x:x.split(';'))
#    c1=c1.apply(lambda x:len(x))
#    
#    data[['predict_category_property']]=c1
#    
#    #########时间戳时间格式化
#    
#    c2=data[['context_timestamp']]
#    
#    c2=c2['context_timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))
#    
#    c3=c2.astype(str).apply(lambda x:x.split(' '))
#    
#    
#    data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
#    
#    
#    data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))
#    
#    
#    
#    del data['context_timestamp']
#    
    #data[['context_timestamp']]=c33
    
#    #######根据instance_id进行重复项的删除
#    data=data.drop_duplicates(['instance_id'])
#    
#    data.shape
#    
#    data.dtypes
    
    
    
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
    
    
    
    
    
    ###########合并另外提取的特征
    new_feature_data=data.drop(['instance_id','context_id'],axis=1,inplace=False)
    
    
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
    return new_feature_data



data21=data1[(data1.day<24)]
data22=data1[(data1.day==24)]

new_train_data=data_process(data)

#new_train_data=pd.merge(new_train_data,item_is_trade,on='item_id',how='left')
#
#new_train_data=pd.merge(new_train_data,item_brand_trade,on='item_brand_id',how='left')
#new_train_data=pd.merge(new_train_data,user_trade,on='user_id',how='left')
#new_train_data=pd.merge(new_train_data,page_trade,on='context_page_id',how='left')
#new_train_data=pd.merge(new_train_data,shop_id_trade,on='shop_id',how='left')
#new_train_data=pd.merge(new_train_data,user_item_trade,on=['user_id','item_id'],how='left')
#new_train_data=pd.merge(new_train_data,user_item_brand_trade,on=['user_id','item_brand_id'],how='left')
#new_train_data=pd.merge(new_train_data,user_shop_trade,on=['user_id','shop_id'],how='left')
#
#new_train_data=pd.merge(new_train_data,user_page_trade,on=['user_id','context_page_id'],how='left')
#new_train_data=pd.merge(new_train_data,user_brand_shop_trade,on=['user_id','item_brand_id','shop_id'],how='left')
#new_train_data=pd.merge(new_train_data,brand_page_trade,on=['item_brand_id','context_page_id'],how='left')
#new_train_data=pd.merge(new_train_data,page_shop_trade,on=['context_page_id','shop_id'],how='left')
#
#new_train_data=new_train_data.fillna(0)






##################对测试数据进行同样操作


############
a1=data_24[['item_category_list']]
a11=pd.DataFrame(np.zeros((len(a1),1)))
a12=pd.DataFrame(np.zeros((len(a1),1)))
a1=a1['item_category_list'].apply(lambda x:x.split(';'))

a11=a1.apply(lambda x :x[0])
a12=a1.apply(lambda x :x[1])
a11=pd.DataFrame(a11)
a12=pd.DataFrame(a12)
a12=a12['item_category_list'].apply(lambda x:int(x))

####令商品的category——list为a12
data_24[['item_category_list']]=a12
######广告商品类目表数值化
        
######广告商品属性列表数值化
b1=data_24[['item_property_list']]
b1=b1['item_property_list'].apply(lambda x:x.split(';'))
b1=b1.apply(lambda x:len(x))

data_24[['item_property_list']]=b1

###########上下文信息
####查询词预测类目属性表数值化
c1=data_24[['predict_category_property']]

c1=c1['predict_category_property'].apply(lambda x:x.split(';'))
c1=c1.apply(lambda x:len(x))

data_24[['predict_category_property']]=c1

#
#c2=data[['context_timestamp']]
#c2=c2['context_timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))
#c3=c2.astype(str).apply(lambda x:x.split(' '))
#data['day']=c3.apply(lambda x:x[0]).apply(lambda x:int(x[8:10]))
#data['hour']=c3.apply(lambda x:x[1]).apply(lambda x:int(x[0:2]))
#
#del data['context_timestamp']



#data[['context_timestamp']]=c33

########根据instance_id进行重复项的删除
#data=data.drop_duplicates(['instance_id'])  
#data.shape  
#data.dtypes
################挖掘其他隐含特征
#u=data[['item_id']]
#u.drop_duplicates(inplace=True)


####用户在同一时间段的浏览次数
u1=data_24[['user_id','hour']]
u1['user_hour_see']=1
u1=u1.groupby(['user_id','hour']).agg('sum').reset_index()

user_hour_see=u1

###用户浏览谋商品品牌的时间间隔
u1=data_24[['user_id','item_brand_id','hour']]
u1['user_brand_hour']=1
u1=u1.groupby(['user_id','item_brand_id','hour']).agg('sum').reset_index()

user_brand_hour_see=u1

data_24=pd.merge(data_24,user_hour_see,on=['user_id','hour'],how='left')
data_24=pd.merge(data_24,user_brand_hour_see,on=['user_id','item_brand_id','hour'],how='left')
data_24.fillna(0)

###商品浏览总次数

new_test_data=data_process(data_24)


#new_test_data=pd.merge(new_test_data,item_is_trade,on='item_id',how='left')
#
#new_test_data=pd.merge(new_test_data,item_brand_trade,on='item_brand_id',how='left')
#new_test_data=pd.merge(new_test_data,user_trade,on='user_id',how='left')
#new_test_data=pd.merge(new_test_data,page_trade,on='context_page_id',how='left')
#new_test_data=pd.merge(new_test_data,shop_id_trade,on='shop_id',how='left')
#new_test_data=pd.merge(new_test_data,user_item_trade,on=['user_id','item_id'],how='left')
#new_test_data=pd.merge(new_test_data,user_item_brand_trade,on=['user_id','item_brand_id'],how='left')
#new_test_data=pd.merge(new_test_data,user_shop_trade,on=['user_id','shop_id'],how='left')
#
#new_test_data=pd.merge(new_test_data,user_page_trade,on=['user_id','context_page_id'],how='left')
#new_test_data=pd.merge(new_test_data,user_brand_shop_trade,on=['user_id','item_brand_id','shop_id'],how='left')
#new_test_data=pd.merge(new_test_data,brand_page_trade,on=['item_brand_id','context_page_id'],how='left')
#new_test_data=pd.merge(new_test_data,page_shop_trade,on=['context_page_id','shop_id'],how='left')
#
#new_test_data=new_test_data.fillna(0)










train=new_train_data[(new_train_data.day<24)].drop(['day','is_trade'],axis=1,inplace=False)
test=new_test_data[(new_test_data.day==24)].drop(['day','is_trade'],axis=1,inplace=False)
y_train=new_train_data[(new_train_data.day<24)][['is_trade']]
y_test=new_test_data[(new_test_data.day==24)][['is_trade']]




string2=train.columns.values.tolist()
print(string2)


string3=['item_id','item_category_list','item_brand_id','item_city_id','user_id','user_gender_id','user_occupation_id','context_page_id','shop_id']

import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV


#turn_params=[{'objective':['binary'],'learning_rate':[0.01,0.03,0.05,0.1],'n_estimators':[60,80,100],'max_depth':[6,7,8]}]
#
#clf=GridSearchCV(lgb.LGBMClassifier(seed=7),turn_params,scoring='roc_auc')
#clf.fit(train,y_train)
#print('best params of lgb is:',clf.best_params_)

clf=lgb.LGBMClassifier(num_leaves=50,max_depth=8,learning_rate=0.1,n_estimators=180)
clf.fit(train,y_train,feature_name=string2,categorical_feature=string3)



y_pre=clf.predict_proba(test)[:,1]

print(log_loss(y_test,y_pre))



###########合并另外提取的特征


#data_test=pd.read_csv('round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)
#
#new_data_test=data_precoss(data_test)



#new_feature_data=data.drop(['instance_id','context_id'],axis=1,inplace=False)
#
#
#new_feature_data=pd.merge(new_feature_data,item_feature,on='item_id',how='left')
#new_feature_data=pd.merge(new_feature_data,item_brand_feature,on='item_brand_id',how='left')
#new_feature_data=pd.merge(new_feature_data,user_feature,on='user_id',how='left')
#new_feature_data=pd.merge(new_feature_data,page_feature,on='context_page_id',how='left')
#new_feature_data=pd.merge(new_feature_data,shop_feature,on='shop_id',how='left')
#new_feature_data=pd.merge(new_feature_data,user_item_feature,on=['user_id','item_id'],how='left')
#
#
#new_feature_data=pd.merge(new_feature_data,user_brand_feature,on=['user_id','item_brand_id'],how='left')
##new_feature_data=pd.merge(new_feature_data,user_time_feature,on=['user_id','context_timestamp'],how='left')
#new_feature_data=pd.merge(new_feature_data,user_shop_feature,on=['user_id','shop_id'],how='left')
#new_feature_data=pd.merge(new_feature_data,user_page_feature,on=['user_id','context_page_id'],how='left')
#new_feature_data=pd.merge(new_feature_data,user_brand_shop_feature,on=['user_id','item_brand_id','shop_id'],how='left')
#new_feature_data=pd.merge(new_feature_data,user_brand_page_feature,on=['user_id','item_brand_id','context_page_id'],how='left')
#new_feature_data=pd.merge(new_feature_data,brand_page_feature,on=['item_brand_id','context_page_id'],how='left')
#new_feature_data=pd.merge(new_feature_data,page_shop_feature,on=['context_page_id','shop_id'],how='left')
#return new_feature_data

