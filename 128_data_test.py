# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:44:05 2018

@author: 1
"""

import pandas as pd

data=pd.read_csv('round1_ijcai_18_test_a_20180301.txt',delimiter=' ',header=0)

data.shape

a1=data[['item_category_list']]

import numpy as np
a11=pd.DataFrame(np.zeros((len(a1),1)))
a12=pd.DataFrame(np.zeros((len(a1),1)))
a1=a1['item_category_list'].apply(lambda x:x.split(';'))

a11=a1.apply(lambda x :x[0])
a12=a1.apply(lambda x :x[1])
a11=pd.DataFrame(a11)
a12=pd.DataFrame(a12)

####令商品的category——list为a12
data[['item_category_list']]=a12

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




a13=c31
t=a13
t=pd.DataFrame(t)
t['instance_id_count']=1
t=t.groupby('context_timestamp').agg('sum').reset_index()







