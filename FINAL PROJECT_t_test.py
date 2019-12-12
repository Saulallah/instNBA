#!/usr/bin/env python
# coding: utf-8

# In[58]:


#Name: Nikhil Wora, Vince Natividad, Saul Sallah, Robert York
#Date: 2019-12-19
#Assignment: Final Project Submission


# In[50]:


import pandas as pd
import seaborn as sns
import scipy as sp


# In[51]:

df = pd.read_csv("NBA-Playerdata-2019.csv")


# In[52]:


# Using scipy package for T-test:
from scipy import stats
df_age = df['Age']
df_usg = df['USG%']
stats.ttest_ind(df_age, df_usg)   


# In[53]:


#Using another package for T-test
from scipy.stats import ttest_ind, ttest_ind_from_stats
a = df['Age']
b = df['USG%']
t, p = ttest_ind(a, b, equal_var=False)
print("ttest_ind:            t = %g  p = %g" % (t, p))


# In[55]:




# In[56]:


sns.boxplot(x = 'Age', y = 'USG%', data=df).set_title('Age and Usage Boxplot ')


# In[57]:


df.groupby('Age')[['USG%', 'MP', 'GS', 'TOV']].agg(['min', 'mean', 'max', 'median'])








