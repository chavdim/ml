#does softmax regression on iris flower dataset
#data is in the form: 
#numpy array,
#[[x1, x2, x3, x4, y]   
#        ....        ]    X  150 rows
# y is converted to [1,0,0] form for training and testing with softmax regression
# y has 3 classes {1,2,3}
#using tensorflow
#tutorial used: 
#https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/beginners/index.html

import numpy as np
import tensorflow as tf
import random

#parameters
numParameters = 4
numClasses = 3
trainIterations = 500
splitRatio = 0.2
random.seed(1)
################################################################
#util
################################################################
#changes data y from floats to [1,0,0] form
#returns numpy.array with y data in [1,0,0] form
def changeToZeroOne(dataY,numOfClasses):
	z = np.zeros((dataY.shape[0],3))	
	for i in range(dataY.shape[0]):
		r = np.zeros(numClasses)
		r[int(dataY[i])] = 1
		z[i]=r
	return z
#splits data in train and test set 
#returns train,test numpy.arrays
def splitTrainAndTest(data,outputIndexes,ratio):
	#calculate data of each class
	numClasses = {}
	#used for indexing
	numSample = 0
	for i in data:
		for ii in outputIndexes:
			output = i[ii] 
			if output in numClasses.keys():
 				numClasses[output][0] += 1
 				numClasses[output][1].append(numSample)
 			else:
 				numClasses[output] = [1,[numSample]]
 		numSample += 1
 	#
 	dataCopy = data[:]
 	testSet = np.empty(5, dtype=float)
 	#change needed for each class accourding to ratio
 	#change to a random sample fo required length
 	for i in numClasses:
 		neededForClass = round(numClasses[i][0] * ratio)
 		numClasses[i] = [neededForClass,
 		random.sample(numClasses[i][1],int(neededForClass))]
 	#remove from dataCopy to create train set
 	#add to testSet to create test set
 	offset = 0 
 	for i in numClasses:
 		for ii in numClasses[i][1]:
 			if offset==0:
 				testSet[0:] = data[ii]
 			else:
 				testSet = np.vstack([testSet, data[ii]])
 			dataCopy = np.delete(dataCopy, (ii-offset), axis=0)
 			offset+=1

 	return dataCopy,testSet
################################################################
#prepare data
################################################################
allData = np.genfromtxt('irisNumeric1.data', delimiter=',')
traind , testd = splitTrainAndTest(allData,[4],ratio = splitRatio)
trainX = traind[0:-1,0:numParameters] #columns 0:4 for all rows
trainY = traind[0:-1,numParameters:] #columns 4 for all rows

testX = testd[0:-1,0:numParameters] #columns 0:4 for all rows
testY = testd[0:-1,numParameters:] #columns 4 for all rows

# convert trainY to [1,0,0] form
trainY = changeToZeroOne(trainY,numClasses)
# convert testY to [1,0,0] form
testY = changeToZeroOne(testY,numClasses)

################################################################
#tensor flow variables and model
################################################################
x = tf.placeholder(tf.float32, [None, numParameters])
y = tf.placeholder(tf.float32, [None, numClasses])
#
W = tf.Variable(tf.zeros([numParameters, numClasses]))
b = tf.Variable(tf.zeros([numClasses]))
#
ymodel = tf.nn.softmax(tf.matmul(x, W) + b)
#
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(ymodel), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

################################################################
#run session
################################################################
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for i in range(trainIterations):
		sess.run(train_step, feed_dict={x: trainX, y: trainY})
		
	#print accuracy
	correct_prediction = tf.equal(tf.argmax(ymodel,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print "Accuracy on test set: ",sess.run(accuracy, feed_dict={x: testX, y: testY})



