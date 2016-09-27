#####################################################################################
## Lab Question 1
#####################################################################################
import os
os.chdir('C:/Users/Salvador.Nunez/GitHub/DAT210x/Module6')

import pandas as pd

## Load up the /Module6/Datasets/parkinsons.data
## data set into a variable X, being sure to drop the name column.
X = pd.read_csv('Datasets/parkinsons.data')
X.drop('name', axis = 1, inplace = True)


## Splice out the status column into a variable y and delete it from X
y = X.status
X.drop('status', axis = 1, inplace = True)

## Perform a train/test split. 30% test group size, with a random_state equal to 7.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


## Create a SVC classifier. Don't specify any parameters, just leave everything as default.
## # Fit it against your training data and then score your testing data.
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print score


#####################################################################################
## Lab Question 2
#####################################################################################

import pandas as pd
import numpy as np

## Load up the /Module6/Datasets/parkinsons.data
## data set into a variable X, being sure to drop the name column.
X = pd.read_csv('Datasets/parkinsons.data')
X.drop('name', axis = 1, inplace = True)


## Splice out the status column into a variable y and delete it from X
y = X.status
X.drop('status', axis = 1, inplace = True)

## Perform a train/test split. 30% test group size, with a random_state equal to 7.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


## Create a SVC classifier. Don't specify any parameters, just leave everything as default.
## # Fit it against your training data and then score your testing data.
from sklearn.svm import SVC


best_score = 0
for i in np.arange(start=0.05, stop=2.05, step=0.05):
    for j in np.arange(start=0.001, stop=0.101, step=0.001):
        model = SVC(C=i, gamma=j)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_C = model.C
            best_gamma = model.gamma
print "The highest score obtained:", best_score
print "C value:", best_C
print "gamma value:", best_gamma


#####################################################################################
## Lab Question 3
#####################################################################################

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

Xo = pd.read_csv('Datasets/parkinsons.data')
Xo.drop('name', axis = 1, inplace = True)

y = Xo.status
Xo.drop('status', axis = 1, inplace = True)

Xs = []
Xs.append(Xo)
Xs.append(preprocessing.StandardScaler().fit_transform(Xo))
Xs.append(preprocessing.MinMaxScaler().fit_transform(Xo))
Xs.append(preprocessing.Normalizer().fit_transform(Xo))
Xs.append(preprocessing.scale(Xo))

best_score = 0

for p in xrange(5):

    X = Xs[p]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

    for i in np.arange(start=0.05, stop=2.05, step=0.05):
        for j in np.arange(start=0.001, stop=0.101, step=0.001):
            model = SVC(C=i, gamma=j)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_C = model.C
                best_gamma = model.gamma
                best_pre_processing = p

print "The highest score obtained:", best_score
print "C value:", best_C
print "gamma value:", best_gamma
print "pre-processing option:", best_pre_processing



#####################################################################################
## Lab Question 4
#####################################################################################

import os
os.chdir('C:/Users/Salvador.Nunez/GitHub/DAT210x/Module6')

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.manifold import Isomap
from sklearn.cross_validation import train_test_split

Xo = pd.read_csv('Datasets/parkinsons.data')
Xo.drop('name', axis = 1, inplace = True)

y = Xo.status
Xo.drop('status', axis = 1, inplace = True)

Xs = []
Xs.append(Xo)
Xs.append(preprocessing.StandardScaler().fit_transform(Xo))
Xs.append(preprocessing.MinMaxScaler().fit_transform(Xo))
Xs.append(preprocessing.Normalizer().fit_transform(Xo))
Xs.append(preprocessing.scale(Xo))

best_score = 0

for p in xrange(5):

    T = Xs[p]

    for k in range(2, 6):

        for l in range(4, 7):

            iso = Isomap(n_neighbors=k, n_components=l)
            X = iso.fit_transform(T)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

            for i in np.arange(start=0.05, stop=2.05, step=0.05):

                for j in np.arange(start=0.001, stop=0.101, step=0.001):

                    model = SVC(C=i, gamma=j)
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)

                    if score > best_score:
                        best_score = score
                        best_C = model.C
                        best_gamma = model.gamma
                        best_pre_processing = p
                        best_n_neighbors = iso.n_neighbors
                        best_n_components = iso.n_components

print "The highest score obtained:", best_score
print "C value:", best_C
print "gamma value:", best_gamma
print "pre-processing option:", best_pre_processing
print "isomap n_neighbors:", best_n_neighbors
print "isomap n_components:", best_n_components
