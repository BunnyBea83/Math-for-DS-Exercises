#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# # A1)Import the data and create two new columns. Create one column that is the number of years since 1790. Create another column that is the population in millions.

# In[2]:


#Import data frame
df = pd.read_csv('us_pop_data.csv')

#Create new column of years since 1790
df['Years_Since_1790']= df['year']-1790

#Create a column for population in millions
df['Population_in_Millions'] = df['us_pop']/1e6  #divide by 1 million to convert to millions

# Display new dataframe
df.head()


# # (B1) Plot the US population (in millions) versus the years since 1790

# In[3]:


# plot the data
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(df["Population_in_Millions"], df["Years_Since_1790"], color= 'purple')
plt.xlabel('US Population (in million)')
plt.ylabel('Years since 1790')
plt.title('US Population Over Years')
plt.show()


# # (C1) Create a linear regression model to predict the US population (in millions) t years from 1790. Find and report the R2-value of this model.

# In[4]:


#Create Regression line with SKlearn
reg = linear_model.LinearRegression()
reg.fit(df[["Population_in_Millions"]], df[["Years_Since_1790"]])


# In[5]:


#Define Values
x = df[['Population_in_Millions']]
y = df [['Years_Since_1790']]
y_pred = reg.predict(x)
m = reg.coef_
b = reg.intercept_


#Define the r2
R2 = r2_score(y,y_pred)
print (f'{R2:.0%} of the variation in y is explained by the variation in x.')

# Plot the new line
plt.plot(df["Population_in_Millions"], df["Years_Since_1790"], color= 'purple')
plt.plot(x,y_pred, color = 'red')
plt.xlabel('US Population (in million)')
plt.ylabel('Years since 1790')
plt.title('US Population Over Years')



# # (D1) Create another new column in your data by squaring the number of years since 1790.

# In[6]:


df['Years^2']= df['Years_Since_1790']**2
df.head()


# # E1) Run another linear regression, where your input feature is the square of the number of years since 1790. Find and report the R2-value of this model.

# In[7]:


#Create Regression line with SKlearn
reg.fit(df[["Population_in_Millions"]], df[["Years^2"]])


# In[8]:


#Create ne prediction and define the r4
y_pred2 = reg.predict(x)
R4 = r2_score(y,y_pred2)
print (f'{R4:.0%} of the variation in y is explained by the variation in x.')

# Plot the new lines
plt.plot(df["Population_in_Millions"], df["Years_Since_1790"], color= 'purple')
plt.plot(x,y_pred, color = 'red')
plt.plot(x,y_pred2, color = 'blue')
plt.xlabel('US Population (in million)')
plt.ylabel('Years since 1790')
plt.xlim(0,400)
plt.ylim(0,1000)
plt.title('US Population Over Years')


# # (f) Plot the models you built on top of the data. Which one fits the data better? Is this apparent in your R2-values. Explain.

# In[9]:


plt.plot(df["Population_in_Millions"], df["Years_Since_1790"], color= 'green', label='True Data')
plt.plot(x,y_pred, color = 'red', label = 'Predicted Data')
plt.plot(x,y_pred2, color = 'blue', label = 'Squared Data')
plt.xlabel('US Population (in million)')
plt.ylabel('Years since 1790')
plt.xlim(0,400)
plt.ylim(0,1000)
plt.title('US Population Over Years')
plt.legend()

print (f'{R2:.0%} of the variation in the predicted y is explained by the variation in x.')
print (f'{R4:.0%} of the variation in the squared predicted y is explained by the variation in x.')


# ## The first predicted set of data fits the coefficient of determination better. This is because this set of data is only 8% unexplained. Whereas the other set of data is unexplained by the millions.

# In[ ]:





#  #  Customer Spending Data
#  ## (A2) Make a histogram of the customer spending amounts

# In[10]:


#Import data frame
df2 = pd.read_csv('customer_spending.csv')
df2.head()


# In[11]:


df2.plot.hist(bins=100)


# # # (b2) Make a new data set that is a log transformation of the customer spending amounts

# In[12]:


log_trans=np.log(df2['ann_spending'])


# In[13]:


df2.head()


# ## (c2) Make a histogram of the log transformed dataset.

# In[14]:


plt.hist(log_trans, bins= 20 )
df2.plot.hist(bins=100)


# # (d2) Compare the two histograms. Discuss why it might be useful to apply a log transformation to this data for modeling purposes.

# ### The purpose of a log transformation is to reduce the amount of skewness in the data set. If we look at the original dataset, there are large fluxuations in the data. While it does have a general bell curve, there are still divits in the curve. By adding the log transformation, this creates a clear bell curve visual that will make it easier for steakholders to digest and comprehend.

# In[ ]:




