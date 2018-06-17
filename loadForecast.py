
# coding: utf-8

# # Load Forecast
# ## SVM Regression
# 

# In[1]:

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read load data
load_data = pd.read_csv("rsfenergymodel2011.csv")
print "data loaded"


# ### Implementation: Data Exploration
# 
# 

# In[2]:

n_total_data = len(load_data)

# number of features
# column B, C, E, F, G, H
n_features = 6 # need to revise based on later coding

# count number of data for Jan, Feb, Mar
n_Jan = len(load_data[load_data['Month'] == 1])
n_Feb = len(load_data[load_data['Month'] == 2])
n_Mar = len(load_data[load_data['Month'] == 3])

n_Winter = n_Jan+n_Feb+n_Mar

# Print the results
print "Total data: {}".format(n_total_data)
print "Number of features: {}".format(n_features)
print "n_Jan: {}".format(n_Jan)
print "n_Feb: {}".format(n_Feb)
print "n_Mar: {}".format(n_Mar)
print "n_Winter: {}".format(n_Winter)


# ## Preparing the Data
# 
# ### Identify feature and target columns
# 

# In[3]:

# Extract feature columns
#feature_cols = ['Weekday', 'Month', 'Day', 'Hour', 'Outside Wet-Bulb Temp (F)', 
           # 'Outside Dry-Bulb Temp (F)', ]
feature_cols_1 = list(load_data.columns[1:3])
feature_cols_2 = list(load_data.columns[4:8])

feature_cols = feature_cols_1+feature_cols_2
target_cols = list(load_data.columns[8:17]) 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)

print "\nTarget column: {}".format(target_cols)

X_all = load_data[feature_cols]
Y_cols = load_data[target_cols]

Y_all = Y_cols.sum(axis=1)
print "\nFeature values:"
print X_all.head()
print "\ntarget values:"
print Y_all.head()


# In[4]:

X_winter = X_all[0:2160]
Y_winter = Y_all[0:2160]

print "\nFeature values:"
print X_winter.head()
print "\ntarget values:"
print Y_winter.head()


# ### Preprocess Feature Columns
# 

# In[5]:

def preprocess_features(X):
    #''' Preprocesses the data'''
    #'''Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        if col_data.dtype == object:
            col_data = col_data.replace(['Monday','Tuesday','Wednesday','Thursday','Friday',
                                        'Saturday','Sunday'], [1, 2, 3, 4, 5, 60, 70])

        output = output.join(col_data)
    
    return output

X_winter = preprocess_features(X_winter)

print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
print X_winter.head()
print Y_winter.head()


# ### Implementation: Training and Testing Data Split
# 
# - Randomly shuffle and split the data (`X_winter`, `Y_winter`) into training and testing subsets.
#   - 75% data for training
#   - 25% data for testing

# In[6]:

#load lib
from sklearn.cross_validation import train_test_split

# Set 75% data for training
num_train = 1620

# Set 75% data for testing
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test= train_test_split(X_winter, Y_winter, train_size=num_train, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
print X_train.head()
print y_train.head()


# ## Training and Evaluating Models
# 
# - Support Vector Regression (SVR)
# 

# In[ ]:

from sklearn.svm import SVR
import matplotlib.pyplot as plt

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)


# #############################################################################
# Look at the results
##

a = np.arange(0, 540, 1)
lw = 2
print len(a)
print len(y_rbf)

plt.scatter(a, y_test, color='darkorange', label='data')

plt.plot(a, y_rbf, color='c', lw=lw, label='RBF model')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

