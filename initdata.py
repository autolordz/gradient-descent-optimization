# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:57:59 2019

@author: autol
"""

#%%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,ShuffleSplit
from sklearn.datasets import load_boston

def init_data(n=20,rstat=12334):
#    rstat = 45
    r = np.random.RandomState(rstat)
    x = np.linspace(-1,1,n);x
    noise = r.uniform(size=n);noise
    X = np.vstack([x**2,np.sin(x)]).T;X
    y = np.sin(x+noise);y
    #y = X.dot(np.ones(2))+noise;y
    X,y = shuffle(X,y,random_state=rstat);X
    return X,y

def init_data1(n=20,rstat=12334,w=np.ones(2),b=12345):
    wn = len(w)
    np.random.seed(rstat)
    X = np.random.random(size=(n, wn))
    y = X.dot(w) + np.random.normal(size=n) + b #+ np.random.randint(0,b,n)
    X,y = shuffle(X,y,random_state=rstat)
    return X,y

def data_b(X,):
    X_b = np.hstack([X, np.ones((len(X), 1))])
    return X_b

def init_data_boston():
    X,y = load_boston(return_X_y=1)
    return X,y

def init_data_house(n=20,rstat=12334,w=np.ones(2)):
    df = pd.read_csv('kc_house_data.csv').dropna()
    df = df[:n]
#    ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
#       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
#       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
#       'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    #w_col = list(filter(lambda x:'sqft'in x, list(df.columns)))
    cols = ['sqft_living','bedrooms', 'bathrooms','sqft_lot', 'floors', 'waterfront']
    X,y = get_numpy_data(df,cols[:len(w)],'price')
#    X,y = get_numpy_data1(df,['sqft_living','bedrooms'],'price')
    X,y = shuffle(X,y,random_state=rstat)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#    X,y = X_train,y_train
#    X_t,y_t = X_test,y_test
    return X,y

#def init_data(n = 100,random_state=12334):
#
#    r = np.random.RandomState(random_state)
#    x = np.linspace(0,1,n)
##    noise = np.random.uniform(size=n)
#    noise = r.uniform(size=n)
#    #X = np.vstack([25*x,x**5,np.exp(x),np.sin(x),np.tanh(x)]).T
#    #X = np.vstack([np.sin(x),np.tanh(x),np.exp(x)]).T
#    X = np.vstack([25*x,x**5]).T
#    y = np.sin(x+noise)
##    y_noise = y*noise+noise
#    #y = y_noise - y_noise.mean()
#    X,y = shuffle(X,y,random_state=random_state)
#    X,X_t,y,y_t = train_test_split(X, y, test_size=0.2, random_state=random_state)
#    w = np.ones(X.shape[1])
##    w_col = ['w'+str(s) for s in range(len(w))]
#    return X,y,w

#@accepts(X_tag=list)
def get_numpy_data(df, X_tag, y_tag):
    return df[X_tag].to_numpy(),df[y_tag].to_numpy()