#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# importing datasets
# 'feature' dataset consist of the main data through which we will predict the target variable 
# The 'power' dataset consist of 2 columns from which 1 column is the Power column which is the target feature for the feature dataset.
feature=pd.read_csv('features.csv')
power=pd.read_csv('power.csv')


# In[3]:


# merging 2 dataset using inner join and on the Timestamp column as it is the common column
# By merging the 2 datasets we will get 136730 rows
df=pd.merge(feature,power,how='inner',on='Timestamp')


# In[4]:


#set_option is used to display all the columns in a dataset
pd.set_option('display.max_columns',None)
df.sample(5)


# In[5]:


df.shape


# In[6]:


#Timestamp column is not an important feature in predicting the power, therefore it has been dropped
df_train=df.drop('Timestamp',axis=1)


# In[7]:


df_train.head()


# In[8]:


df_train.info()


# In[9]:


df_train.isnull().sum()


# In[10]:


df_train.isnull().sum().sum()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Creating a box plot for each column in the DataFrame
for column in df_train.columns:
    plt.figure()  # Create a new figure for each box plot
    df.boxplot(column=column)
    plt.title('Box Plot - {}'.format(column))
    plt.show()


# The above box plot of each column shows that there are outliers of higher value that is much greater than the mean value of the data points in a column. Therefore to remove outliers we will use the 'describe' function.

# In[12]:


#to understand the data and finding outliers i have used the describe function
df_train.describe()


# As we can see from the above table that the max value of each and every column is '99999' which can be termed as outliers. 

# Before removing outliers we will check histogram of a column for our convinience

# In[13]:


#through the below visualization, it can be seen that the data points are much more between 0 to 20000.
plt.hist(df_train['Temperature Ambient'])
plt.show()


# Now we will check the number of data points which contains the value '99999'. Just for example i will take a particular row which is 'Gearbox_T1_High_Speed_Shaft_Temperature'.

# In[14]:


# there are total 1121 rows which contains 99999
df_train.query('Gearbox_T1_High_Speed_Shaft_Temperature==99999.0')


# In[15]:


hist_range=(0,100)
sns.histplot(df_train['Gearbox_T1_High_Speed_Shaft_Temperature'], binrange=hist_range)


# In[16]:


plt.scatter(df_train['Converter Control Unit Reactive Power'],df_train['Power(kW)'])
# plt.xlim([0,])
#the values above 90000 are outliers


# In[17]:


#replacing the value '99999' with Nan 
for column in df_train.columns:
    df_train[column] = df_train[column].replace(99999, np.nan)


# In[18]:


df_train.describe()


# In[19]:


plt.scatter(df_train['Converter Control Unit Reactive Power'],df_train['Power(kW)'])


# In[20]:


df_train.isnull().sum().sum()


# In[21]:


# Count the number of values greater than the specific value in each column
greater_than_count = (df_train > 800).sum().sum()

# Count the number of values less than the specific value in each column
less_than_count = (df_train < 800).sum().sum()
print(greater_than_count)
print(less_than_count)


# In[22]:


#replacing null values by their mean
for i in df_train.columns:
    df_train[i] = df_train[i].fillna(df_train[i].mean())


# In[23]:


df_train.isnull().sum()


# In[24]:


#for more outlier detection i will use IQR 
import pandas as pd
import numpy as np

lower_bound = 0.25  # Lower bound quartile
upper_bound = 0.75  # Upper bound quartile
threshold = 1.5  # Threshold for outliers

# Function to identify outliers and replace them with NaN
def replace_outliers(column):
    q1 = column.quantile(lower_bound)
    q3 = column.quantile(upper_bound)
    iqr = q3 - q1
    outliers = (column < (q1 - threshold * iqr)) | (column > (q3 + threshold * iqr))
    return column.where(~outliers, other=np.nan)

# Apply the outlier detection function to each column
df_test = df_train.apply(replace_outliers)
#-----------------------------------------------------------------------------------------------
# import random

# # Iterate over each column in the DataFrame
# for column in df_test.columns:
#     # Get the non-null values from the column
#     non_null_values = df_test[column].dropna()
    
#     # Get the count of NaN values in the column
#     null_count = df_test[column].isnull().sum()
    
#     # Generate random values from the non-null values
#     random_values = random.choices(non_null_values, k=null_count)
    
#     # Replace the NaN values with random values in the column
#     df_test[column].fillna(pd.Series(random_values), inplace=True)


# Now the values above 800 in some columns would be removeeed as it can be seen as an outlier.
# After removing the outliers the data has much Nan values. Therefore we should replace it with the mean values.

# Mean/Median/Mode Imputation: Replace the NaN values with the mean, median, or mode of the respective column. This approach works well for numerical data and can be easily implemented using the fillna() method.
# 
# Forward or Backward Fill: Propagate the last known value forward or the next known value backward to fill the NaN values. This method is useful for time series or sequential data where the order of values is important. You can use the fillna() method with the method parameter set to "ffill" (forward fill) or "bfill" (backward fill).
# 
# Interpolation: Estimate the missing values based on the existing values using interpolation techniques such as linear interpolation, polynomial interpolation, or spline interpolation. The interpolate() method in pandas provides options for different interpolation methods.

# In[25]:


import pandas as pd

# Perform interpolation to fill NaN values
df_interpolated = df_test.interpolate()

# Display the DataFrame with interpolated values
df_interpolated


# In[26]:


df_interpolated.isnull().sum()


# In[27]:


df_interpolated.isnull().sum().sum()


# In[28]:


df_test=df_interpolated


# In[29]:


#finding correlation between the features
corr=df_test.corr()


# In[30]:


a=corr.iloc[76,:]


# In[31]:


a


# In[32]:


#finding top 20 features
#having less features gets us a good accuracy
#by performing replacement of outliers
a.nlargest(21)


# In[33]:


#to know the distribution of power
sns.displot(x=df_test[r'Power(kW)'])


# In[34]:


# performed lasso regression


# In[35]:


#selecting most important features in the dataset
columns_to_select = ['Torque', 'Gearbox_T1_Intermediate_Speed_Shaft_Temperature', 'Gearbox_T1_High_Speed_Shaft_Temperature','Gearbox_T3_High_Speed_Shaft_Temperature','Gearbox_Oil-2_Temperature','Tower Acceleration Lateral','Temperature Gearbox Bearing Hollow Shaft','Gearbox_T3_Intermediate_Speed_Shaft_Temperature','Gearbox_Oil-1_Temperature','Gearbox_Oil_Temperature','Gearbox_Oil_Temperature','Temperature Bearing_A','Temperature Trafo-3','Voltage A-N','Tower Acceleration Normal','Temperature Trafo-2','Voltage C-N','Converter Control Unit Reactive Power','Reactive Power','Converter Control Unit Voltage','Power(kW)']


# In[36]:


#cloning the df_test
df_test1=df_test[columns_to_select]


# In[37]:


X = df_test1.drop('Power(kW)', axis=1)
y = df_test1['Power(kW)']


# In[38]:


from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


# Create a Lasso regression model
lasso = Lasso(alpha=0.6)  # You can adjust the alpha value as needed


# In[41]:


# Fitting the Lasso model to the training data
lasso.fit(X_train, y_train)


# In[42]:


# Making predictions on the testing data
y_pred = lasso.predict(X_test)


# In[43]:


# Evaluate accuracy of the model using R-squared score
accuracy = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Accuracy (R-squared score):", accuracy)
print(mse)
print(mae)


# In[44]:


#gradient boosting regressor


# In[45]:


from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load the dataset
X = df_test1.drop('Power(kW)', axis=1)
y = df_test1['Power(kW)']

# Perform PCA on the data
n_components = 10  # Specified the desired number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create and train a Gradient Boosting Regressor model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Accuracy (R-squared score):", accuracy)
print(mse)
print(mae)


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt

df = df_test1

# box plot for each column in the DataFrame
for column in df.columns:
    plt.figure()  # figure for each box plot
    df.boxplot(column=column)
    plt.title('Box Plot - {}'.format(column))
    plt.show()


# In[47]:


#Random Forrest Regressor


# In[48]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df_test1.drop('Power(kW)', axis=1)
y = df_test1['Power(kW)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("Mean Absolute Error (MAE): ", mae)
print("R-squared (R2): ", r2)


# In[49]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

X = df_test1.drop('Power(kW)', axis=1)
y = df_test1['Power(kW)']


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Printing the evaluation metrics for training and testing datasets
print("Training Set:")
print("Mean Squared Error (MSE): ", mse_train)
print("Mean Absolute Error (MAE): ", mae_train)
print("R-squared (R2): ", r2_train)
print()
print("Testing Set:")
print("Mean Squared Error (MSE): ", mse_test)
print("Mean Absolute Error (MAE): ", mae_test)
print("R-squared (R2): ", r2_test)

# Plot predicted values vs actual values for both training and testing datasets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Set')
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Testing Set')
plt.tight_layout()
plt.show()


# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df_test1.drop('Power(kW)', axis=1)
y = df_test1['Power(kW)']
# Split the data into training and validation/test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the training data
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Evaluate the model on the validation/test data
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)

print("Training MSE:", train_mse)
print("Validation MSE:", val_mse)
r2 = r2_score(y_train, y_train_pred)
print(r2)


# In[51]:


#Saving the Random Forrest Model


# In[52]:


import pickle
with open('final_RF_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)


# In[53]:


with open('final_RF_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[54]:


#saving the Linear Regression model


# In[55]:


import pickle
with open('final_LR_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[56]:


with open('final_LR_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[57]:


#saving the Lasso model


# In[58]:


import pickle
with open('final_Lasso_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[59]:


with open('final_Lasso_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[ ]:





# In[ ]:





# In conclusion, 
# the accuracy of the Random forrest is 99.6%,
# the accuracy of Lasso Regression is 89.1%,
# the accuracy of the Gradient boosting is 96.4%.

# In[61]:


import ast

with open('C:\Users\91812\Rishabh_Project\Wind Turbine Power Prediction.ipynb', 'r') as f:
    notebook_content = f.read()

notebook = ast.literal_eval(notebook_content)

used_libraries = set()

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for line in source:
            if line.startswith('import') or line.startswith('from'):
                library = line.split()[1]
                used_libraries.add(library)

print(used_libraries)


# In[ ]:


Wind Turbine Power Prediction.ipynb


# In[62]:


import pkg_resources

installed_packages = pkg_resources.working_set
for package in installed_packages:
    print(package.key, package.version)


# In[ ]:





# In[ ]:





# In[ ]:




