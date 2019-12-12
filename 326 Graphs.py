#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Name: Nikhil Wora
#Directory ID: 114763136
#Date: December 09, 2019
#Assignment: Graph Coding


# In[15]:


#Import Python Libraries
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Show graphs within Python notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Read in csv file
df = pd.read_csv('/Users/NikhilWoraMacPro/Desktop/INST 326/PYs/NBA-Playerdata-2019.csv')

# Use package to draw a histogram
sns.distplot(df['Age'])

#Add title
plt.title('Is age a significant factor towards Usage Rate %?')
plt.xlabel('Age')
plt.ylabel('USG %')


# In[21]:


#Scatterplot graph
# Age of player * usage rate
sns.jointplot(x='Age', y='USG%', data=df)

#add labels
plt.title('What age has the biggest usage in the NBA?')

