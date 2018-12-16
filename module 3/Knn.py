
# coding: utf-8

# In[6]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


iris=load_iris()
x=iris.data
y=iris.target


knn=KNeighborsClassifier(n_neighbors = 5)
score = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
print(score)


# In[7]:


print(score.mean())


# In[10]:


k_range = range(1, 31)
k_score= []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores=cross_val_score(knn,x,y,cv=10,scoring = 'accuracy')
    k_score.append(scores.mean())
print(k_score)


# In[12]:


plt.plot(k_range, k_score)
plt.xlabel('value of K for KNN')
plt.ylabel('cross validation accuracy')

