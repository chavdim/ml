#Performs single output regression on the iris flower dataset.
#Classifies between 3 different iris flower types.

#source ~/tensorflow/bin/activate 
import pandas
import skflow
import random
import numpy as np
from numpy import genfromtxt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

random.seed(42) 
#
#data is in the form x1,x2,x3,x4,y
dataTrain = genfromtxt('irisNumeric1.data', delimiter=',')
#trainX = x1,x2,x3,x4 for all rows
trainX = dataTrain[1:,:4]
#trainT = y for all rows
trainY = dataTrain[1:,4:]

#print trainX.shape
#print trainY.shape



#####
#####Splits into test and train sets
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
#####
#####
regressor = skflow.TensorFlowDNNRegressor(
	n_classes=3,
    hidden_units=[5],  
    batch_size=128, 
    steps=2000, 
    learning_rate=0.1)

regressor.fit(X_train, y_train)
#score=mean_absolute_error(regressor.predict(X_test), y_test)
score=mean_squared_error(regressor.predict(X_test), y_test)

print score