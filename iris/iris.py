import numpy as np
import tensorflow as tf
import random


numParameters = 4
numClasses = 3
trainIterations = 1000
splitRatio = 0.2
random.seed(1)
#
#if wchich class = 3 , will return numpy.array([0,0,3])
#
def classToZeroOne(whichClass,numberOfClasses):
	r= np.zeros(numberOfClasses)
	r[whichClass] = 1
	return r
#
#splits data in train adn test set 
#returns train,test nparrays
#
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
 	#testSet = np.empty((30,1))
 	testSet = np.empty(5, dtype=float)

 	#change needed for each class accourding to ratio
 	#change to a random sample fo required length
 	for i in numClasses:
 		neededForClass = round(numClasses[i][0] * ratio)
 		numClasses[i] = [neededForClass,
 		random.sample(numClasses[i][1],int(neededForClass))]
 	#
 	offset = 0 
 	for i in numClasses:
 		for ii in numClasses[i][1]:
 			if offset==0:
 				
 				testSet[0:] = data[ii]
 				#print testSet[0:]
 			else:
 				testSet = np.vstack([testSet, data[ii]])
 			#testSet.append(data[ii])
 			#print data[ii].shape, testSet.shape
 			#testSet = np.vstack([testSet, data[ii]])
 			#print data[ii]
 			#print "ts-------"
 			#print testSet
 			#print "------------------"
 			#testSet = np.append(testSet,data[ii],0)
 			dataCopy = np.delete(dataCopy, (ii-offset), axis=0)
 			offset+=1
 			
 	
 	return dataCopy,testSet


#
#prepare data
#
allData = np.genfromtxt('irisNumeric1.data', delimiter=',')
traind , testd = splitTrainAndTest(allData,[4],ratio = splitRatio)
trainX = traind[0:-1,0:numParameters] #columns 0:4 for all rows
trainY = traind[0:-1,numParameters:] #columns 4 for all rows

testX = testd[0:-1,0:numParameters] #columns 0:4 for all rows
testY = testd[0:-1,numParameters:] #columns 4 for all rows
#testY.astype(float)
#testX.astype(float)

# convert trainY to numClasses columns 
z = np.zeros((trainY.shape[0],3))

for i in range(trainY.shape[0]):
	r = np.zeros(numClasses)
	r[int(trainY[i])] = 1
	z[i]=r
	
trainY = z
# convert testY to numClasses columns 
z = np.zeros((testY.shape[0],3))

for i in range(testY.shape[0]):
	r = np.zeros(numClasses)
	r[int(testY[i])] = 1
	z[i]=r
	
testY = z
print trainX,trainY
print testX,testY



#
#tensor flow variables and model
#
x = tf.placeholder(tf.float32, [None, numParameters])
y = tf.placeholder(tf.float32, [None, numClasses])
#
W = tf.Variable(tf.zeros([numParameters, numClasses]))
b = tf.Variable(tf.zeros([numClasses]))
#
ymodel = tf.nn.softmax(tf.matmul(x, W) + b)
#
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(ymodel), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



#
#run session
#
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for i in range(trainIterations):
		#batch_xs, batch_ys = mnist.train.next_batch(100)

		#print trainX,trainY
		sess.run(train_step, feed_dict={x: trainX, y: trainY})
		

	correct_prediction = tf.equal(tf.argmax(ymodel,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print sess.run(accuracy, feed_dict={x: testX, y: testY})
