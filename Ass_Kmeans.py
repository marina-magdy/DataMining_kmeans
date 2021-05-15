#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from pylab import *
import seaborn as sns


# In[25]:


df = pd.read_csv('./Mall_Customers.csv' ,index_col='CustomerID')


# In[26]:


df.head()


# In[27]:


df.isnull().sum()


# In[36]:


df.info()


# In[37]:


df.describe()


# In[32]:


x = df.iloc[:,[2,3]].values
print(x)


# In[33]:


wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters = i, init='k-means++' , random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[34]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1,11) ,wcss, color='red')
plt.title('elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()


# In[35]:


kmeans= KMeans(n_clusters=5, init='k-means++' , random_state=42)
y_pred=kmeans.fit_predict(x)
plt.figure(figsize=(10,6))
for i in range(5):
    plt.scatter(x[y_pred==i,0] , x[y_pred==i,1])


# In[ ]:




