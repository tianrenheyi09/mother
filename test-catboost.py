# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:30:52 2018

@author: 1
"""

from catboost import CatBoostClassifier
cat_features=[0,1]
train_data=[["a","b",1,4,5,6],["a","b",4,5,6,7],["c","d",30,40,50,60]]
train_labels=[1,1,-1]
test_data=[["a","b",2,4,6,8],["a","d",1,4,50,60]]
model=CatBoostClassifier(iterations=2,learning_rate=1,depth=2,loss_function='Logloss')
model.fit(train_data,train_labels,cat_features)

preds_class=model.predict(test_data)
preds_proba=model.predict_proba(test_data)
preds_raw=model.predict(test_data,prediction_type='RawFormulaVal')
