#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np # linear algebra
import pandas as pd # data processing


# In[114]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[115]:


data = pd.read_csv('marketing_campaign.csv', delimiter='\t')
data


# In[116]:


data.info()


# In[117]:


data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])


# In[118]:


data.describe().T


# In[120]:


data.isna().sum()


# In[119]:


#Z_costContact and Z_Revenue are having zero variance so we can exclude them
data.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)


# In[121]:


#Removing rows having null income values
data = data[data['Income'].notnull()]
data.reset_index(drop=True, inplace=True)
data


# In[122]:


data['Year'] = data['Dt_Customer'].apply(lambda row: row.year)
data


# In[123]:


data['Age'] = data['Year'] - data['Year_Birth']
data.drop(['Dt_Customer'],axis=1,inplace=True)


# In[125]:


vis_data = data.copy()
vis_data


# In[126]:


vis_data.drop(['ID','Year_Birth','Education','Marital_Status','Year'
],axis=1, inplace=True)

vis_data


# In[127]:


sns.distplot(vis_data['Recency'])


# In[129]:


sns.distplot(data['Income'])


# In[130]:


outlier_idx = vis_data[vis_data['Income'] > 150000].index
vis_data.drop(outlier_idx, inplace=True)

sns.distplot(vis_data['Income'])


# In[131]:


sns.distplot(data['Age'])


# In[132]:


outlier_age = vis_data.loc[vis_data['Age'] > 90].index
vis_data.drop(outlier_age, inplace=True)
vis_data.reset_index(drop=True, inplace=True)

sns.distplot(vis_data['Age'])


# In[133]:


sns.countplot(vis_data['Kidhome'])


# In[134]:


sns.countplot(vis_data['Teenhome'])


# In[149]:


vis_data['Kidhome'] = vis_data['Kidhome'].apply(lambda row: 1 if row >= 1 else 0)
vis_data['Teenhome'] = vis_data['Teenhome'].apply(lambda row: 1 if row >= 1 else 0)

vis_data


# In[150]:


sns.countplot(vis_data['NumDealsPurchases'])


# In[153]:


sns.countplot(vis_data['Response'])


# In[138]:


from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, Isomap
from sklearn.preprocessing import StandardScaler

c_al_data = vis_data.copy()
c_al_data


# In[139]:


num_col = ['Income', 'Age', 'NumDealsPurchases', 'NumWebVisitsMonth']
cat_col = ['Kidhome', 'Teenhome', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']


# In[140]:


c_al_data['Accepted'] = c_al_data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1) > 0
c_al_data['Accepted'] = c_al_data['Accepted'].apply(lambda row: 1 if row else 0)

c_al_data.drop(['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'], axis=1, inplace=True)

c_al_data


# In[151]:


c_al_data = pd.DataFrame(StandardScaler().fit_transform(c_al_data), columns=c_al_data.columns)
c_al_data


# In[142]:


for i in c_al_data.columns:
    c_al_data[i].astype(dtype=float)

c_al_data.info()


# In[143]:


from sklearn.decomposition import PCA

pca = PCA(n_components=10)

pca_data = pd.DataFrame(pca.fit_transform(c_al_data))

plt.plot(pca.explained_variance_ratio_.cumsum())

c_al_data = pd.concat([c_al_data, pca_data], axis=1)

c_al_data


# In[144]:


tsne = TSNE(learning_rate=50)

tsne_results = c_al_data.copy()

tsne_results[['TSNE1', 'TSNE2']] = pd.DataFrame(tsne.fit_transform(c_al_data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), columns=['TSNE1', 'TSNE2'])

tsne_results


# In[145]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Accepted', data=tsne_results)


# In[146]:


from scipy.cluster.vq import kmeans, vq
import random

random.seed(1000)

# All Data
cluster_data = c_al_data.copy()

distortions = []
for k in range(1, 15):
    _, distortion = kmeans(cluster_data, k)
    distortions.append(distortion)
plt.plot(distortions, label='All Data')

# Some Meaningful only
cluster_data = c_al_data.copy()[[
    'Kidhome', 'Teenhome', 'Complain', 'Accepted'
]]

distortions = []
for k in range(1, 15):
    _, distortion = kmeans(cluster_data, k)
    distortions.append(distortion)    
plt.plot(distortions, label='Home Only')

# PCA Only
cluster_data = c_al_data.copy()[[
    0, 1, 2, 3, 4, 5, 6, 7
]]

distortions = []
for k in range(1, 15):
    _, distortion = kmeans(cluster_data, k)
    distortions.append(distortion)
    
plt.plot(distortions, label='PCA Only')




plt.legend()


# In[147]:


random.seed(500)

cluster_data = c_al_data.copy()[['Kidhome', 'Teenhome', 'Complain', 'Accepted']]

cluster_centers, _ = kmeans(cluster_data, 5)

cluster_data['cluster_labels'], _ = vq(cluster_data, cluster_centers)

tsne = TSNE(learning_rate=100)

tsne_results = cluster_data.copy()

tsne_results[['TSNE1', 'TSNE2']] = pd.DataFrame(tsne.fit_transform(c_al_data), columns=['TSNE1', 'TSNE2'])

plt.figure(figsize=(10,10))

sns.scatterplot(x='TSNE1', y='TSNE2', hue='cluster_labels', cmap=sns.color_palette(), data=tsne_results)


# In[148]:


random.seed(500)

cluster_data = c_al_data.copy()[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

cluster_centers, _ = kmeans(cluster_data, 5)

cluster_data['cluster_labels'], _ = vq(cluster_data, cluster_centers)

tsne = TSNE(learning_rate=100)

tsne_results = cluster_data.copy()

tsne_results[['TSNE1', 'TSNE2']] = pd.DataFrame(tsne.fit_transform(c_al_data), columns=['TSNE1', 'TSNE2'])

plt.figure(figsize=(10,10))

sns.scatterplot(x='TSNE1', y='TSNE2', hue='cluster_labels', cmap=sns.color_palette(), data=tsne_results)


# In[ ]:





# In[ ]:




