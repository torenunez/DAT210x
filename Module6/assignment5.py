import pandas as pd


#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names

import os
os.chdir('C:/Users/Salvador.Nunez/GitHub/DAT210x/Module6')
# 
# Load up the mushroom dataset into dataframe 'X'
names = ['classes', 'cap-shape', 'cap-surface',
         'cap-color', 'bruises?', 'odor', 'gill-attachment',
         'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
         'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
         'stalk-color-above-ring', 'stalk-color-below-ring',
         'veil-type', 'veil-color', 'ring-number', 'ring-type',
         'spore-print-color', 'population', 'habitat']
X = pd.read_csv('Datasets/agaricus-lepiota.data', names = names, na_values = '?')
# Verify you did it properly.
# Indices shouldn't be doubled.
print X.head()

# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
# INFO: An easy way to show which rows have nans in them
print X[pd.isnull(X).any(axis=1)]

#
# Go ahead and drop any row with a nan
#
print X.shape
X.dropna(axis=0, how='any', inplace=True)
print X.shape


#
# Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
y = X.classes
X.drop('classes', axis=1, inplace=True)
y = y.map({'e': 0,'p': 1})


#
# Encode the entire dataset using dummies
#
rem_names = names[1:]
X = pd.get_dummies(X, columns = rem_names)
print X.head(6)

# 
# Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=7)



#
# Create an DT classifier. No need to set any parameters
#
from sklearn import tree
model = tree.DecisionTreeClassifier()


# train the classifier on the training data / labels:
model.fit(X_train, y_train)
# score the classifier on the testing data / labels:
score = model.score(X_test, y_test)
# .. your code here ..
print "High-Dimensionality Score: ", round((score*100), 3)


feature_imp = pd.DataFrame(zip(X.columns, model.feature_importances_))
print feature_imp.sort(1, ascending=False)


#
# Use the code on the courses SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz. If you can't, use: http://webgraphviz.com/
#
tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)

from subprocess import call
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])


