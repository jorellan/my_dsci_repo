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
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print "\n"
print("----------K-NEAREST NEIGHBOR-------------------")

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)


# Build a sequence of models for k = 2, 4, 6, 8, ..., 20.
ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
for k in ks:
	# Create model and fit.
	mod = neighbors.KNeighborsClassifier(n_neighbors=k)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print(' EVALUATING MODEL: k = ' + str(k) )
	# Look at results.
	print_binary_classif_error_report(y_test, preds)
	
print("\n----------Bernoulli - NAIVE BAYESIAN-------------------")
gnb_mod = naive_bayes.BernoulliNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_binary_classif_error_report(y_test, preds)
	
print("\n----------DECISION TREE-------------------")
print("DTREE WITH GINI IMPURITY CRITERION:")
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_binary_classif_error_report(y_test, preds_gini)
print("\n")
print('DTREE WITH ENTROPY CRITERION:')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_binary_classif_error_report(y_test, preds_entropy)

print("\n----------RANDOM FOREST-------------------")
n_est = [5, 10, 50, 100]
depth = [.25, .5, .75,  None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp))
		# Look at results.
		print_binary_classif_error_report(y_test, preds)
#print(list(x_test))

print("\n----------SUPPORT VECTOR MACHINE-------------------")
# Build a sequence of models for different n_est and depth values. **NOTE: c=1.0 is equivalent to the default.
cs = [0.2, 0.5, 1.0, 2.0, 2.5]
for c in cs:
	# Create model and fit.
	mod = svm.SVC(C=c)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('EVALUATING MODEL: C = ' + str(c) )
	# Look at results.
	print_binary_classif_error_report(y_test, preds)


