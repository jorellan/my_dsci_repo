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
	
#----Data A ------#
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
#lower the corresponding p-value, the more certain we are of a relationship

#-----------------------Baseline model-----------------------#

# Split training and test sets from main set
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
base_model = linear_model.LinearRegression()
base_model.fit(x_train, y_train) # Fit the model.
preds = base_model.predict(x_test) # Make predictions
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print("BASE MODEL:" "\n"
	'MSE: ' + str(mean_squared_error(y_test, preds)) + "\n"
	'MAE: ' + str(median_absolute_error(y_test, preds)) + "\n"
	'R^2: ' + str(r2_score(y_test, preds)) + "\n"
	"EVS: " + str(explained_variance_score(y_test, preds)))
							   
#---------------Percentile-based feature selection-----------------#

## Create a percentile-based feature selector based on the F-scores. Get top 25% best features by F-test.
selector_f = SelectPercentile(f_regression, percentile=35)
selector_f.fit(x_train, y_train) 

# Get the columns of the best 25% features.	
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)
percent_model = linear_model.LinearRegression()
percent_model.fit(xt_train, y_train) # Fit the model.
preds = percent_model.predict(xt_test) # Predictions on test data 
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print("TOP 25% MODEL:" "\n"
	'MSE: ' + str(mean_squared_error(y_test, preds)) + "\n"
	'MAE: ' + str(median_absolute_error(y_test, preds)) + "\n"
	'R^2: ' + str(r2_score(y_test, preds)) + "\n"
	"EVS: " + str(explained_variance_score(y_test, preds)))

#---------------Recursive Feature Elimination with Cross Validation -----------------#

# Use RFECV to arrive at the approximate best set of predictors. RFECV is a greedy method.
selector_f = RFECV(estimator=linear_model.LinearRegression(), \
                   cv=5, scoring=make_scorer(r2_score))
selector_f.fit(x_train, y_train)

# Get the columns of the best 25% features.	
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)
model = linear_model.LinearRegression() # Create a least squares linear regression model.
model.fit(xt_train, y_train)# Fit the model.
preds = model.predict(xt_test)# Make predictions
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print("RFECV MODEL:" "\n"
	'MSE: ' + str(mean_squared_error(y_test, preds)) + "\n"
	'MAE: ' + str(median_absolute_error(y_test, preds)) + "\n"
	'R^2: ' + str(r2_score(y_test, preds)) + "\n"
	"EVS: " + str(explained_variance_score(y_test, preds)))
	
						#---------------Lasso---------------#

# Show Lasso regression fits for different alphas.
alphas = [ 2.5, 5.0, 10.0, 12, 20, 30, 40, 50, 10 ]
for a in alphas:
	# Normalizing transforms all variables to number of standard deviations away from mean.
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train, y_train)
	preds = lasso_mod.predict(x_test)
	print('Lasso Model with alpha=' + str(a) + ':'+ "\n" + 'R^2:' + str(r2_score(y_test, preds)) + 
	"\n" + 'MSE: ' + str(mean_squared_error(y_test, preds)))
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
lasso_mod = linear_model.Lasso(alpha=10, normalize=True, fit_intercept=True)
lasso_mod.fit(x_train, y_train)	

#----------------------------------------------------------------------------------------#
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

print("\n""\n""BASE MODEL:" "\n"
	'MSE: ' + str(mean_squared_error(data_yb, predsbasic)) + "\n"
	'MAE: ' + str(median_absolute_error(data_yb, predsbasic)) + "\n"
	'R^2: ' + str(r2_score(data_yb, predsbasic)) + "\n"
	"EVS: " + str(explained_variance_score(data_yb, predsbasic)))

print("\n"'Lasso Model with alpha=10: ' + "\n" +	
	'MSE: ' + str(mean_squared_error(data_yb, predslasso)) + "\n"
	'MAE: ' + str(median_absolute_error(data_yb, predslasso)) + "\n"
	'R^2: ' + str(r2_score(data_yb, predslasso)) + "\n"
	"EVS: " + str(explained_variance_score(data_yb, predslasso)))

#pprint.pprint(pd.DataFrame({'Actual':data_yb, 'Base Model Predicted':predsbasic, 'Lasso Model Predicted':predslasso })





