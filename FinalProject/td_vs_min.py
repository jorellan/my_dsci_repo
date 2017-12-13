#Jennifer Orellana#
execfile("pddf.py")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pprint
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score



column = ['Division Name', 'School Name', 'PassRate', 'White', 'Minority', 'Total Disadvantaged']
data_x = data['Total Disadvantaged']
data_y = data['Minority'] 
plt.plot(np.unique(data_x), np.poly1d(np.polyfit(data_x, data_y, 1))(np.unique(data_x)))
plt.scatter(data_x,data_y)
#plt.show()	
data_x = data_x.reshape(-1, 1) #unknowwn #of columns but one row? // data has a single feature

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
							
							   
						   
							   
							   