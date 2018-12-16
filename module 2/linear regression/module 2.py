
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr


from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')
address = '/home/sanyam/Desktop/dataset/enrollment_forecast.csv'
enroll = pd.read_csv(address)
enroll.columns = ['year','roll','unem','hgrad','inc']
enroll.head()


# In[3]:


sb.pairplot(enroll)


# In[5]:


print (enroll.corr())


# In[14]:


enroll_data = enroll.ix[:,(2,3)].values
enroll_target = enroll.ix[:,1].values

enroll_data_names = ['unem','hgrad']
X, y= scale(enroll_data), enroll_target


# In[15]:


missing_values = X == np.NAN
X[missing_values == True]


# In[16]:


LinReg=LinearRegression(normalize = True) 
LinReg.fit(X,y)
print(LinReg.score(X,y))

