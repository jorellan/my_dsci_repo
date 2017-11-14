#Jennifer Orellana#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import pprint
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score




# Get a list of the categorical features for a given dataframe. Move to util file for future use!
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. Move to util file for future use!	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
	
setA = pd.read_csv('./Assignment_2/AmesHousingSetA.csv')

# one-hot encoding.
setA = pd.get_dummies(setA, columns=cat_features(setA))

# Get all non-SalePrice columns as features.
features = list(setA)
features.remove('SalePrice')

# Select x and y data
data_x = setA[features]
data_y = setA['SalePrice'] 

#impute values
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x = imp.fit_transform(data_x)

