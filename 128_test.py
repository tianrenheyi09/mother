

import pandas as pd

data=pd.read_csv('D:/mother/round1_ijcai_18_train_20180301.txt',delimiter=' ',header=0)

data.shape


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

######划分需要进行编码的特征数据

dataset1=data.loc[:,['item_id','item_category_list','item_brand_id','item_city_id','user_id',
               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]



dataset2=data.loc[:,['item_property_list','item_property_list','item_price_level','item_sales_level',
                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
                     'context_page_id','predict_category_property','shop_review_num_level',
                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
                     'shop_score_description']]

label=data.loc[:,'is_trade']




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
from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)

from sklearn.preprocessing import OneHotEncoder


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)


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
                     

X_train_lr1=X_train_lr[['item_id']]
X_train_lr2=X_train_lr[['item_category_list']]
X_train_lr3=X_train_lr[['item_brand_id']]
X_train_lr4=X_train_lr[['item_city_id']]
X_train_lr5=X_train_lr[['user_id']]
X_train_lr6=X_train_lr[['user_gender_id']]
X_train_lr7=X_train_lr[['user_occupation_id']]
X_train_lr8=X_train_lr[['context_timestamp']]
X_train_lr9=X_train_lr[['shop_id']]
X_train_lr10=X_train_lr[['item_property_list','item_property_list','item_price_level','item_sales_level',
                     'item_collected_level','item_pv_level','user_age_level','user_star_level',
                     'context_page_id','predict_category_property','shop_review_num_level',
                     'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
                     'shop_score_description']]

n_estimator = 20

grd_1 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_1 = OneHotEncoder()
grd_lm_1 = LogisticRegression()
grd_1.fit(X_train_1, y_train)

grd_enc_1.fit(grd_1.apply(X_train_1)[:, :, 0])

lr1=grd_enc_1.transform(grd_1.apply(X_train_lr1)[:, :, 0])




grd_2 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_2 = OneHotEncoder()
grd_lm_2 = LogisticRegression()
grd_2.fit(X_train_2, y_train)

grd_enc_2.fit(grd_2.apply(X_train_2)[:, :, 0])

lr2=grd_enc_2.transform(grd_2.apply(X_train_lr2)[:, :, 0])


grd_3 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_3 = OneHotEncoder()
grd_lm_3 = LogisticRegression()
grd_3.fit(X_train_3, y_train)

grd_enc_3.fit(grd_3.apply(X_train_3)[:, :, 0])

lr3=grd_enc_3.transform(grd_3.apply(X_train_lr3)[:, :, 0])


grd_4 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_4 = OneHotEncoder()
grd_lm_4 = LogisticRegression()
grd_4.fit(X_train_4, y_train)

grd_enc_4.fit(grd_4.apply(X_train_4)[:, :, 0])

lr4=grd_enc_4.transform(grd_4.apply(X_train_lr4)[:, :, 0])



grd_5 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_5 = OneHotEncoder()
grd_lm_5 = LogisticRegression()
grd_5.fit(X_train_5, y_train)

grd_enc_5.fit(grd_5.apply(X_train_5)[:, :, 0])

lr5=grd_enc_5.transform(grd_5.apply(X_train_lr5)[:, :, 0])


grd_6 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_6 = OneHotEncoder()
grd_lm_6 = LogisticRegression()
grd_6.fit(X_train_6, y_train)

grd_enc_6.fit(grd_6.apply(X_train_6)[:, :, 0])

lr6=grd_enc_6.transform(grd_6.apply(X_train_lr6)[:, :, 0])



grd_7 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_7 = OneHotEncoder()
grd_lm_7 = LogisticRegression()
grd_7.fit(X_train_7, y_train)

grd_enc_7.fit(grd_7.apply(X_train_7)[:, :, 0])

lr7=grd_enc_7.transform(grd_7.apply(X_train_lr7)[:, :, 0])



grd_8 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_8 = OneHotEncoder()
grd_lm_8 = LogisticRegression()
grd_8.fit(X_train_8, y_train)

grd_enc_8.fit(grd_8.apply(X_train_8)[:, :, 0])

lr8=grd_enc_8.transform(grd_8.apply(X_train_lr8)[:, :, 0])


grd_9 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_9 = OneHotEncoder()
grd_lm_9 = LogisticRegression()
grd_9.fit(X_train_9, y_train)

grd_enc_9.fit(grd_9.apply(X_train_9)[:, :, 0])

lr9=grd_enc_9.transform(grd_9.apply(X_train_lr9)[:, :, 0])



grd_10 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_10 = OneHotEncoder()
grd_lm_10 = LogisticRegression()
grd_10.fit(X_train_10, y_train)

grd_enc_10.fit(grd_10.apply(X_train_10)[:, :, 0])

lr10=grd_enc_10.transform(grd_10.apply(X_train_lr10)[:, :, 0])



X_train_11=X_train[['item_id','item_category_list','item_brand_id','item_city_id','user_id',
               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]
               
X_train_lr11=X_train_lr[['item_id','item_category_list','item_brand_id','item_city_id','user_id',
               'user_gender_id','user_occupation_id','context_timestamp','shop_id']]    

grd_11 = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc_11 = OneHotEncoder()
grd_lm_11 = LogisticRegression()
grd_11.fit(X_train_11, y_train)

grd_enc_11.fit(grd_11.apply(X_train_11)[:, :, 0])

lr11=grd_enc_11.transform(grd_11.apply(X_train_lr11)[:, :, 0])





lr1=pd.DataFrame(lr1.toarray())
lr2=pd.DataFrame(lr2.toarray())
lr3=pd.DataFrame(lr3.toarray())
lr4=pd.DataFrame(lr4.toarray())
lr5=pd.DataFrame(lr5.toarray())
lr6=pd.DataFrame(lr6.toarray())
lr7=pd.DataFrame(lr7.toarray())
lr8=pd.DataFrame(lr8.toarray())
lr9=pd.DataFrame(lr9.toarray())
lr10=pd.DataFrame(lr10.toarray())



result=pd.concat([lr1,lr2,lr3,lr4,lr5,lr6,lr7,lr8,lr9,lr10],axis=1)



#######result为总的GBDT提取出来的特征向量
grd_lm=LogisticRegression()


grd_lm.fit(result, y_train_lr)




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

ltest1=pd.DataFrame(ltest1.toarray())
ltest2=pd.DataFrame(ltest2.toarray())
ltest3=pd.DataFrame(ltest3.toarray())
ltest4=pd.DataFrame(ltest4.toarray())
ltest5=pd.DataFrame(ltest5.toarray())
ltest6=pd.DataFrame(ltest6.toarray())
ltest7=pd.DataFrame(ltest7.toarray())
ltest8=pd.DataFrame(ltest8.toarray())
ltest9=pd.DataFrame(ltest9.toarray())
ltest10=pd.DataFrame(ltest10.toarray())


new_test=pd.concat([ltest1,ltest2,ltest3,ltest4,ltest5,ltest6,ltest7,ltest8,ltest9,ltest10],axis=1)




y_pred_grd_lm = grd_lm.predict_proba(new_test)[:, 1]
    

fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')







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
