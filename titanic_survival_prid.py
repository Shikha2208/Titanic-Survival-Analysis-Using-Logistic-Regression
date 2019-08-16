#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[9]:


titanic_data=pd.read_csv('E:/SHIKHA_FOLDER/logistic regession/titanic/train.csv')


# In[11]:


titanic_data.head(20)


# In[14]:


titanic_data.shape


# In[16]:


print("# of passengers in original data :" +str(len(titanic_data.index)) )


# ## Analysing  data

# In[18]:


sns.countplot(x="Survived", data=titanic_data)


# In[21]:


sns.countplot(x="Survived", hue="Sex",data=titanic_data) 


# In[22]:


sns.countplot(x="Survived", hue="Pclass",data=titanic_data) 


# In[24]:


titanic_data["Age"].plot.hist()


# In[27]:


titanic_data["Fare"].plot.hist()


# In[28]:


titanic_data["Fare"].plot.hist(bins=20 ,figsize=(10,5))


# In[29]:


titanic_data.info()


# In[32]:


sns.countplot(x="SibSp",data=titanic_data)


# ## Data Wrangling

# In[34]:


titanic_data.isnull()


# In[35]:


titanic_data.isnull().sum()


# In[41]:


sns.heatmap(titanic_data.isnull(), yticklabels=False)


# In[45]:


sns.heatmap(titanic_data.isnull(), yticklabels=False,cmap="viridis")


# In[46]:


sns.boxplot(x="Pclass",y="Age",data=titanic_data)


# In[47]:


titanic_data.head(5)


# In[49]:


titanic_data.drop("Cabin", axis=1, inplace=True)


# In[50]:


titanic_data.head(5)


# In[51]:


titanic_data.dropna(inplace=True)


# In[53]:


sns.heatmap(titanic_data.isnull(), yticklabels=False,cbar=False)


# In[54]:


titanic_data.shape


# In[56]:


titanic_data.isnull().sum()


# In[57]:


titanic_data.head(2)


# In[58]:


pd.get_dummies(titanic_data["Sex"])


# In[60]:


sex=pd.get_dummies(titanic_data["Sex"],drop_first="True")


# In[61]:


sex.head(2)


# In[63]:


embark=pd.get_dummies(titanic_data["Embarked"])
embark.head(5)


# In[64]:


embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head(5)


# In[65]:


pcl=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
pcl.head(5)


# In[66]:


titanic_data=pd.concat([titanic_data,sex,embark,pcl],axis=1)


# In[73]:


titanic_data.head(5)


# In[69]:


titanic_data.head(10)


# In[74]:


# this command we usedfordeletingmultiple column
#titanic_data.drop(['Sex',"Embarked",'PassengerId',"Name","Ticket"],axis=1, inplace=True)


# In[75]:


titanic_data.drop(['Pclass'],axis=1, inplace=True)


# In[76]:


titanic_data.head(10)


# ## Training

# In[77]:


X=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]


# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=1)


# In[82]:


from sklearn.linear_model import LogisticRegression


# In[83]:


logmodel=LogisticRegression()


# In[84]:


logmodel.fit(X_train,y_train)


# In[86]:


prediction=logmodel.predict(X_test)


# In[88]:


from sklearn.metrics import classification_report


# In[89]:


classification_report(y_test,prediction)


# In[90]:


from sklearn.metrics import confusion_matrix


# In[91]:


confusion_matrix(y_test,prediction)


# In[92]:


from sklearn.metrics import accuracy_score


# In[93]:


accuracy_score(y_test,prediction)


# In[ ]:




