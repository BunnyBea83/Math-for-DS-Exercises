#!/usr/bin/env python
# coding: utf-8

# # Exercise 2.2
# 
# ## 1) Make a scatterplot of the highway miles per gallon (y-axis) versus the weight (x-axis).

# In[2]:


#Import Python DLC's
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


# Import car data, display data
df = pd.read_csv('car_data.csv')
df


# In[4]:


#Import an Scatterplot of the data with "weight"=x and "Highway mpg"=y, Implement stars as the markers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Weight')
plt.ylabel('Highway MPG')
plt.scatter(df.weight,df.hwy_mpg, color = 'orange',marker= '*')


# ## 2.) Based on your plot, what is the general trend of how highway miles per gallon varies with the weight?

# ### Based on my plot, the general trend of how car mpg vary with weight is that the heavier the car is the less MPG it will produce
# 
# ### Your dad's Ford F-150 could never beat your Toyota Camry in a MPG competition because the Ford F-150 is atleast 1.5 x heavier than the Camry.

# ## 3.) If you were to build a linear model using this data to predict highway miles per gallon from weight, would you expect the slope to be positive or negative? Explain.

# ### Using this data to build a linear model I would expect to slope to also be negative.  Reason being, If this data is to be refleted as our "sample size" the sample should reflect the same direction/value as the larger sample. Meaning that the sample is the same as the linear model, it's just a smaller version. A close up of the whole picture.

# ## 4) If the slope of a linear model predicting highway miles per gallon from the weight, interpret the meaning of the slope being −0.05.
# 
# ### If the slope of the cars mile is represented by -0.05 that would mean that the relationship between MPG and weight is negative. For every increase in the value of weight there will be a decrease in the value of MPG by 0.05 units.

# ## 5.) Write code to add a line to the graph you made in problem (1). Adjust the slope and y-intercept of this line until you think you have found the line that best fits the data. Record the slope and y-intercept.

# In[5]:


#Create new x and y values
m= -0.01
b= 58
pred_x= [min(df.weight),max(df.weight)]
pred_y= [m*xi+b for xi in pred_x]

#Plot line of regression with new x and y values
plt.plot(pred_x,pred_y,color='red')

#Import an Scatterplot of the data with "weight"=x and "Highway mpg"=y, Implement stars as the markers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Weight')
plt.ylabel('Highway MPG')
plt.scatter(df.weight,df.hwy_mpg, color = 'orange',marker= '*')


# ## 6) Use Python to find the best-fit line. The Scikit-learn package is a good choice to use for this.

# In[93]:


reg = linear_model.LinearRegression()
reg.fit(df[['weight']],df[['hwy_mpg']])


# In[94]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Weight')
plt.ylabel('Highway MPG')
plt.scatter(df.weight,df.hwy_mpg, color = 'orange',marker= '*')
plt.plot(df.weight,reg.predict(df[['weight']]), color ='blue')


# In[97]:


# Create scatter plot and Regrassion line
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Weight')
plt.ylabel('Highway MPG')
plt.scatter(df.weight,df.hwy_mpg, color = 'orange',marker= '*')

#add regression line using .predict
plt.plot(X,reg.predict(X),color='blue')


# ## 7. Find the root mean squared error (RMSE) of the prediction line you found in problem (4) and the actual best-fit line found in problem (5). How do these compare?

# ### I'm assuming you mean problem 5 and 6, not 4 and 5

# In[147]:


#import features
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Define values for scatter plot in problem 5
m= -0.01
b= 58
X = df[['weight']]
Y= df[['hwy_mpg']]

#Create values for predicted y
def Y_pred(m,X,b):
    return m*X+b

#Calculate RMSE
RMSE2= (math.sqrt(np.square(np.subtract(Y,np.array(Y_pred(m,X,b))).mean())))
print (f' The RMSE for the Linear Regression in problem 5 is {RMSE2:,.2f}.')
    


# Define values for scatter plot in problem 6
x = df[['weight']]
y = df[['hwy_mpg']]
y_pred = reg.predict(df[['weight']])

#calculate RMSE
RMSE= math.sqrt(np.square(np.subtract(y,y_pred)).mean())
print(f' The RMSE for Linear Regression in problem 6 is {RMSE:,.2f}.')


# ### These compare by my problem 5 regression being a lesser value compared to the best-fit regression. The smaller the number is, the closer it should be to its accuracy. So theoretically, my line should be more acurate than the best-fit line.

# ## 8) Use the best-fit line in problem (5) to predict the highway mpg of a car that weighs 3200 pounds.

# In[107]:


#find m
m = reg.coef_

#find b
b = reg.intercept_

#set value for x
x=3200

#calculate
y= int(m*x+b)
print(f'If a car weighs 3200 lbs then it runs on {y} mpg')


# In[ ]:




