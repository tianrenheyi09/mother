# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:56:06 2018

@author: 1
"""




import pandas as pd
new_feature_data=pd.read('new_feature_data.csv',header=0)


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
