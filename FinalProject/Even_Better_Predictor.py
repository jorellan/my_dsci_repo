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

#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#
#--------------------------------2017 Scores---------------------------------------------#
#----------------------------------------------------------------------------------------#						  
score17 = pd.read_csv('./TestScores17.csv')
score17 = score17.drop([ 'Sch Type', 'Low\rGrade', 'High\rGrade', 'Subject'],axis=1)
score17.PassRate = pd.to_numeric(score17.PassRate, errors='coerce')
score17 = score17.groupby(['Div Name','School Name']).mean().reset_index()
score17.columns = ['Division Name', 'School Name', 'PassRate']
#print (list(score17))
#print (score17.head(5))

##Race
ethnicity17 = pd.read_csv('./2017 Ethnicity.csv') 
ethnicity17 = ethnicity17.drop(['Division No.', 'School No.',  'Grade', 'Total\rFull-time Students', 'Part-time Students',  'Hispanic', 'American_Indian_Alaska_Native', 'Asian', 'Black', 'Native_Hawaiian_Pacific_Islander','Two_More_Races'],axis=1)
ethnicity17['Totals'] = pd.to_numeric(ethnicity17['Totals'], errors='coerce')
ethnicity17 = ethnicity17.groupby([ 'Division Name',"School Name"]).sum().reset_index()
ethnic_per = ethnicity17.copy()
ethnic_per["White"] = (ethnicity17["White"]/ethnicity17['Totals'])
ethnic_per["Minority"] = (1-ethnic_per["White"])
ethnic_per = ethnic_per.drop("Totals", axis=1)
#print(list(ethnic_per))
#print(ethnic_per.head(5))


##Financial Well Being
ethnicity17 = ethnicity17.drop("White", axis=1)
economic17 = pd.read_csv('./2017 Disadvantaged.csv') 
delete = ['Div. No.', 'School No.','Low Grade', 'High Grade', 'Grade PK', 'Grade JK', 'Grade KG', 'Grade T1', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12', 'Grade PG', 'Total Full-Time', 'Part-time Students']
economic17 = economic17.drop(delete, axis=1)
economic17.columns = ['Division Name', 'School Name', 'Total Disadvantaged']
economic17['Total Disadvantaged'] = pd.to_numeric(economic17['Total Disadvantaged'], errors='coerce')
economic = pd.merge(ethnicity17, economic17, on=['Division Name', "School Name"])
economic_per = economic
economic_per['Total Disadvantaged'] = (economic['Total Disadvantaged']/economic['Totals'])
economic_per = economic_per.drop("Totals", axis=1)
#print(list(economic_per))
#print(economic_per.head(5))

#----------Data------------#
data2 = pd.merge(score17, economic_per, on=['Division Name', 'School Name'])	
data2 = pd.merge(data2, ethnic_per,  on=['Division Name', "School Name"])
features = ['Total Disadvantaged', 'Minority', 'PassRate']
data2 = data2[features]
features = list(data2)
features.remove("PassRate")
#print(list(data2))
#print(data2.head)

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
add_missing_dummy_columns(data2,features)	

data_x2 = data2[features]
data_y2 = data2['PassRate'] 
print(len(data_x))
print(len(data_x2))

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0) #impute
data_x2 = imp.fit_transform(data_x2)
print(data_x2)

# Make predictions on test data and look at the results.
preds = model.predict(data_x2)
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MAE: ' + str(median_absolute_error(data_y2, preds)) )
print('R2: '  +  str(r2_score(data_y2, preds)))

						   
			