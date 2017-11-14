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
from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
execfile("ModelBuilding.py")

#----DataB Prediction-------#

setB = pd.read_csv('./Assignment_2/AmesHousingSetB.csv')

# one-hot encoding.
setB = pd.get_dummies(setB, columns=cat_features(setB))

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
add_missing_dummy_columns(setB, features)

# Select x and y data
data_xb = setB[features]
data_yb = setB['SalePrice'] 

#impute values
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_xb = imp.fit_transform(data_xb)

#----------------------------------------------------------------------------------------#
predsbasic = base_model.predict(data_xb) # Make basic predictions 
predslasso = lasso_mod.predict(data_xb) # Make Lasso predictions


print("\n\n""BASE MODEL:" "\n"
	'MSE: ' + str(mean_squared_error(data_yb, predsbasic)) + "\n"
	'MAE: ' + str(median_absolute_error(data_yb, predsbasic)) + "\n"
	'R^2: ' + str(r2_score(data_yb, predsbasic)) + "\n"
	"EVS: " + str(explained_variance_score(data_yb, predsbasic)))

print("\n"'Lasso Model with alpha=10: ' + "\n" +	
	'MSE: ' + str(mean_squared_error(data_yb, predslasso)) + "\n"
	'MAE: ' + str(median_absolute_error(data_yb, predslasso)) + "\n"
	'R^2: ' + str(r2_score(data_yb, predslasso)) + "\n"
	"EVS: " + str(explained_variance_score(data_yb, predslasso)))

pprint.pprint(pd.DataFrame({'Actual':data_yb, 'Base Model Predicted':predsbasic, 'Lasso Model Predicted':predslasso }))
