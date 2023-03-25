#!/usr/bin/env python
# coding: utf-8

# # California Housing Regression Machine Learning

# In[1]:


import pandas as pd
import numpy as np
from numpy import mean
import lightgbm as lgb
from lightgbm import LGBMRegressor
from verstack import LGBMTuner
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer,StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_california_housing


# Read input data

# In[2]:


train = pd.read_csv(r'./input_data/train_extended.csv')
test = pd.read_csv(r'./input_data/test.csv')


# ### Assign the Isolation Forest model

# In[3]:


clf = IsolationForest(contamination =0.05,max_samples=0.7 ,random_state=0).fit(train)
OD = clf.predict(train.values)
Outlier_rows = []


# Delete outliers

# In[4]:


for i in range(train.shape[0]):
    if OD[i] == -1:
        Outlier_rows.append(i)
train = train.drop(Outlier_rows)
train = train.reset_index(drop=True)
train.drop(train.columns[0],axis=1,inplace=True)
train.head()


# Drop test id column

# In[5]:


test.drop(['id'],axis=1,inplace=True)
test.head()


# ### Triggering the LGBMTuner
# 
# This module tunes the model automatically. For getting stable prediction, we run the LGBMTuner multiple times and return their mean as the final prediction.
# 

# In[6]:


def stable_prediction(n_trials):
    
    predictions = pd.DataFrame(columns = [i for i in range(n_trials)])
    
    for trial in range(n_trials):

        X = train.values[:,:-1]
        Y = train.values[:,-1]
        
        # the only required argument
        tuner = LGBMTuner(metric = 'rmse',trials = 30,seed = 13)
        #tuner = LGBMTuner(metric = 'rmse',trials = 150,seed = 13)

        #the tuner needs these datatype for X and Y
        X = pd.DataFrame(X)
        Y = pd.Series(Y)
        tuner.fit(X,Y)
        test_df = pd.DataFrame(test.values[:,:-1])
        predicted = tuner.predict(test_df)

        predictions[trial] = predicted
        
    Mean_Prediction = []
    
    for i in range(predictions.shape[0]):
        
        row = predictions.iloc[i].values.tolist()
        Mean = mean(row)
        Mean_Prediction.append(Mean)
    
    return Mean_Prediction,predictions


# In[7]:


Pre = stable_prediction(n_trials=6)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




