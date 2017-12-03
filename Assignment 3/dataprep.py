import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#load data
data = pd.read_csv('./data/churn_data.csv') 
# Get a list of the categorical features for a given dataframe.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))


pd.plotting.scatter_matrix(data, diagonal='kde')
#plt.show()

#print(list(data))

# Select x and y data
features = list(data)
features.remove('CustID')
features.remove('Churn')
data_x = data[features]
data_y = data['Churn']

# one-hot encoding
data_x = pd.get_dummies(data_x, columns=cat_features(data_x))
#print(list(data_x))

# Convert the different class labels to unique numbers with label encoding.
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

