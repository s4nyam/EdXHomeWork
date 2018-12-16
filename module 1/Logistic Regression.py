
# coding: utf-8

# In[4]:


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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')
address = '/home/sanyam/Desktop/dataset/mtcars.csv'
cars=pd.read_csv(address)
cars.columns = ['cars_names','mpg','cyl','disp','hp','qsec','drat','wt','vs','am','gear','carb']
cars.head()


# In[5]:


cars_data = cars.ix[:,(5,11)].values
cars_data_names = ['drat','carb']


y=cars.ix[:,9].values
sb.regplot(x='drat',y='carb', data = cars,scatter = True)


# In[6]:


drat = cars['drat']
carb = cars['carb']
spearmanr_coefficient, p_value=spearmanr(drat, carb)
print("Spearmanr Rank correlation coefficient",0.3 *(spearmanr_coefficient))


# In[7]:


cars.isnull().sum()


# In[11]:


sb.countplot(x='am', data=cars, palette = 'hls')


# In[12]:


cars.info()


# In[15]:


X= cars_data
LogReg=LogisticRegression()
LogReg.fit(X,y)
print(LogReg.score(X, y))


# In[16]:


y_pred = LogReg.predict(X)
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))

