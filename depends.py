# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:55:17 2019

@author: autol
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit,train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

def get_numpy_data(df, X_tag, y_tag):
    df['b_x'] = 1 # this is how you add a constant column to an SFrame
    X = df[['b_x'] + X_tag].to_numpy()
    y = df[y_tag].to_numpy()
    return X,y

def get_numpy_data1(df, X_tag, y_tag):
    X = df[X_tag].to_numpy()
    y = df[y_tag].to_numpy()
    return X,y

def iters_plot(x,y,pl,t,xl='iters',yl='Δw'):
    ''' print iters data '''
    plt.figure()
    plt.loglog(x,y,'o-',label='λ1 = %s'%pl)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(t)
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()
