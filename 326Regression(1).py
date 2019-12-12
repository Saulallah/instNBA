#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Name: Sulayman Sallah
#Directory ID: 114724113
#Date: December 09, 2019
#Assignment: Regression Analysis
import unittest
import pandas as pd
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm 
get_ipython().run_line_magic('matplotlib', 'inline')
def getData():
# Read in CSV file
    nba = pd.read_csv("/Users/Saul/Documents/NBA-Playerdata-2019.csv")

# Variable statistics
    print(nba['Age'].describe())
    print(nba['USG%'].describe())
    print(nba['MP'].describe())
    print(nba['FGA'].describe())
    print(nba['FTA'].describe())
# Read in CSV file
nba = pd.read_csv("/Users/Saul/Documents/NBA-Playerdata-2019.csv")

# Variable statistics
print(nba['Age'].describe())
print(nba['USG%'].describe())
print(nba['MP'].describe())
print(nba['FGA'].describe())
print(nba['FTA'].describe())

#training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)

#Performing regression
age = nba['Age']
usg = nba['USG%']                                                                                 
X = age.values.reshape(-1,1)
Y = usg.values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Making predictions and comparing them to actual results
Y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Usage': Y_test.flatten(), 'Predicted Usage': Y_pred.flatten()})
df

# Making graph showing comparison between 
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Plotting Usage vs Age
nba.plot(x='Age', y='USG%', style='o')  
plt.title('Usage vs Age')  
plt.xlabel('Age')  
plt.ylabel('Usage(%)')  
plt.plot(X_test, Y_pred, color='red', linewidth=2)
plt.show()



#To retrieve the intercept:
print("Intercept:",regressor.intercept_)

#For retrieving the slope:
print("Slope:",regressor.coef_)


X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
# Displyaing regression results
model.summary()


# In[ ]:




