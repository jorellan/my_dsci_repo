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
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble 
from sklearn import svm 
from sklearn.model_selection import train_test_split

execfile("dataprep.py")
def print_binary_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print "\n"

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

print("----------K-NEAREST NEIGHBOR-------------------")
# Build a sequence of models for k = 2, 4, 6, 8, ..., 20.
ks = [16]
for k in ks:
	# Create model and fit.
	mod_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
	mod_knn.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	predsknn = mod_knn.predict(x_test)
	print(' EVALUATING MODEL: k = ' + str(k) )
	# Look at results.
	print_binary_classif_error_report(y_test, predsknn)
	
print("\n----------Bernoulli - NAIVE BAYESIAN-------------------")
bnb_mod = naive_bayes.BernoulliNB()
bnb_mod.fit(x_train, y_train)
predsbnb = bnb_mod.predict(x_test)
print_binary_classif_error_report(y_test, predsbnb)
	
print("\n----------DECISION TREE-------------------")
print("DTREE WITH GINI IMPURITY CRITERION:")
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_binary_classif_error_report(y_test, preds_gini)

print("\n----------SUPPORT VECTOR MACHINE-------------------")
# Build a sequence of models for different n_est and depth values.
cs = [2.0]
for c in cs:
	# Create model and fit.
	mod_svm = svm.SVC(C=c)
	mod_svm.fit(x_train, y_train)
	# Make predictions - both class labels and predicted probabilities.
	predssvm = mod_svm.predict(x_test)
	print('EVALUATING MODEL: C = ' + str(c) )
	# Look at results.
	print_binary_classif_error_report(y_test, predssvm)

print("\n----------RANDOM FOREST-------------------")
n_est = [100]
depth = [None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod_rf = ensemble.RandomForestClassifier(n_estimators=n)
		mod_rf.fit(x_train, y_train)
		# Make predictions - both class labels and predicted probabilities.
		predsrf = mod_rf.predict(x_test)
		print('EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp))
		# Look at results.
		print_binary_classif_error_report(y_test, predsrf)

#----------------------------------------------------------------------------------------#
# Get a list of the categorical features for a given dataframe.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))
	
#load data
new_data = pd.read_csv('./data/churn_validation.csv') 
#print(list(new_data))

# Select x and y data
new_features = list(new_data)
new_features.remove('CustID')
new_features.remove('Churn')
data_nx = new_data[new_features]
data_ny = new_data['Churn']

# one-hot encoding
data_nx = pd.get_dummies(data_nx, columns=cat_features(data_nx))
#print(data_x.count)
#print(data_nx.count)

# Convert the different class labels to unique numbers with label encoding.
le = preprocessing.LabelEncoder()
data_ny = le.fit_transform(data_ny)



#Random Forest
preds = mod_rf.predict(data_nx)
print('PREDICTING with RANDOM FOREST MODEL: n_est = 100, depth = None')
print('Accuracy: ' + str(accuracy_score(data_ny, preds)))
print('F1: ' + str(f1_score(data_ny, preds)))
print('ROC AUC: ' + str(roc_auc_score(data_ny, preds)))
print "\n"

#Illustrate recoding numeric classes back into original (text-based) labels.
y_test_labs = le.inverse_transform(data_ny)
pred_labs = le.inverse_transform(preds)
print('(Actual, Predicted): \n' + str(zip(y_test_labs, pred_labs)))