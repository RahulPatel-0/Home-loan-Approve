#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data =pd.read_csv(r"C:\Users\smara\OneDrive\Desktop\DATA SET\loan_sanction_train.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


from sklearn.preprocessing import LabelEncoder 


# In[9]:


le=LabelEncoder()


# In[10]:


data.head()


# In[11]:


data.Loan_ID=le.fit_transform(data.Loan_ID)


# In[12]:


data.Gender=le.fit_transform(data.Gender)
data.Married=le.fit_transform(data.Married)
data.Dependents=le.fit_transform(data.Dependents)
data.Education=le.fit_transform(data.Education)
data.Self_Employed=le.fit_transform(data.Self_Employed)
data.ApplicantIncome=le.fit_transform(data.ApplicantIncome)
data.CoapplicantIncome=le.fit_transform(data.CoapplicantIncome)
data.LoanAmount=le.fit_transform(data.LoanAmount)
data.Loan_Amount_Term=le.fit_transform(data.Loan_Amount_Term)
data.Credit_History=le.fit_transform(data.Credit_History)
data.Property_Area=le.fit_transform(data.Property_Area)
data.Loan_Status=le.fit_transform(data.Loan_Status)


# In[13]:


data.head()


# In[14]:


data.columns


# In[15]:


y=data['Loan_Status']
x=data[['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']]


# In[16]:


x.head()


# In[17]:


y.head()


# In[ ]:





# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[20]:


x_train


# In[21]:


x_test


# In[22]:


y_train


# In[23]:


y_test


# In[24]:


from sklearn.tree import DecisionTreeClassifier


# In[25]:


model=DecisionTreeClassifier()


# In[31]:


model.fit(x_train,y_train)


# In[33]:


model.score(x_test,y_test)


# In[ ]:





# In[ ]:




