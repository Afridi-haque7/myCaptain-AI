#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


data = pd.read_csv('mnist.csv')


# In[7]:


#viewing column heads
data.head()


# In[8]:


#extracting data from dataset
a = data.iloc[3,1:].values


# In[24]:


#reshaping extracted data
a = a.reshape(28, 28).astype('uint8')
plt.imshow(a)


# In[10]:


#preparing the data
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[11]:


#creating test and train sizes batches
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[12]:


#check data
y_train.head()


# In[16]:


#call rfc
rf = RandomForestClassifier(n_estimators=100)


# In[17]:


#fit the model
rf.fit(x_train, y_train)


# In[18]:


pred = rf.predict(x_test)
pred


# In[19]:


# check prediction accuracy
s = y_test.values
#calculate no of correctly predicted values 
count= 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1
        
count


# In[20]:


len(pred)


# In[21]:


# accuracy value
1895/2000


# In[ ]:




