#Jennifer Orellana#
execfile("pddf.py")
column = ['Division Name', 'School Name', 'PassRate', 'White', 'Minority', 'Total Disadvantaged']
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import pprint
from sklearn import linear_model
from sklearn import preprocessing
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

#----------------------------Disadvantaged & Race Basic----------------------------------------#
data_x = data[['Total Disadvantaged', 'Minority']]
data_y = data['PassRate'] 
sm = pd.plotting.scatter_matrix(data, diagonal='kde')
plt.tight_layout()
#plt.show()	

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)

# Fit the model.
model.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = model.predict(x_test)
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MAE: ' + str(median_absolute_error(y_test, preds)) )
print('R2: '  +  str(r2_score(y_test, preds)))
							
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
	'R^2: ' + str(r2_score(y_test, preds)))
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
	'R^2: ' + str(r2_score(y_test, preds)))

	
#---------------------------------------------Lasso---------------------------------------------#

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
