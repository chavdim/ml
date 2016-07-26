# -*- coding: utf-8 -*-
"""
Created on Sun Oct 05 21:07:06 2014

@author: DD
"""
import gzip,pickle,math
import numpy as np
#np.random.seed(1)

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
small_set=(train_set[0][0:1000,:],train_set[1][0:1000])

f.close()
#classification between nothing, one and minus
#respectively: 0,1,2 
#size in 10 by 10
one=np.zeros((10,10))
one[:,4]=np.array([1,1,1,1,1,1,1,1,1,1])
one[:,5]=np.array([1,1,1,1,1,1,1,1,1,1])

one2=np.zeros((10,10))
one2[:,4]=np.array([1,1,0,1,1,1,1,1,1,1])
one2[:,5]=np.array([1,1,1,1,0,1,1,1,1,1])

minus=np.zeros((10,10))
minus[4]=np.array([1,1,1,1,1,1,1,1,1,1])
minus[5]=np.array([1,1,1,1,1,1,1,1,1,1])

minus2=np.zeros((10,10))
minus2[4]=np.array([1,1,0,1,1,1,1,1,1,1])
minus2[5]=np.array([1,1,1,1,1,0,1,1,1,1])

nothing=np.zeros((10,10))

dataset=np.array([np.reshape(nothing,(1,100))[0],np.reshape(one,(1,100))[0],
                  np.reshape(minus,(1,100))[0], np.reshape(minus2,(1,100))[0],
                  np.reshape(one2,(1,100))[0]])

#util
def sigmoid(x):
    """if x<-200:
        return 0  
    if x>200:
        return 1"""
    return 1 / (1 + math.exp(-x))
# do sigmoid() for all elements in vector
def sV(a):
    #a=np.reshape(np.apply_along_axis(sigmoid, 1, a),(a.shape[0],1))
    b=np.zeros((a.shape[0],1))
    for i in range(a.shape[0]):
        b[i]=sigmoid(a[i])
    return b
def addBiasToArray(a):
    bias=1
    return np.insert(a,0,bias,0)

def getBinaryVector(size,ans):
    r=np.zeros((size,1))
    r[ans]=1
    return r
    
#dataset=np.insert(dataset,0,1,1)
ss=small_set[0]
ss=np.insert(ss,0,1,1)
y=np.array([[0],[1],[2],[2],[1]])
#
class NeuralNetwork():
    def __init__(self,dataset,y,alpha,numMiddleNodes):
        self.ds=dataset
        self.y=y
        self.numInputs=self.ds.shape[1]
        self.k=10
        self.numTrainingExamples=dataset.shape[0]
        self.alpha=alpha
        self.numMiddleNodes=numMiddleNodes
        #
        self.a1=np.array(())
        self.a2=np.array(())
        self.out=np.array(())
        #errors
        self.eOut=np.array(())
        self.eMid=np.array(())
        self.eOutTotal=np.zeros((self.k,1))
        self.eMidTotal=np.zeros((numMiddleNodes,1))
        
        self.theta1Gradient=np.zeros((self.numMiddleNodes,self.numInputs))
        self.theta2Gradient=np.zeros((self.k,self.numMiddleNodes+1))
        
        self.theta1GradientTotal=np.zeros((self.numInputs,self.numMiddleNodes))
        self.theta2GradientTotal=np.zeros((self.numMiddleNodes+1,self.k))
        #
        
        self.addRandomWeigths()
        #self.addOnesAsWeights()
    def addOnesAsWeights(self):
        self.theta1=np.ones((self.numInputs,self.numMiddleNodes))
        self.theta2=np.ones((self.numMiddleNodes,self.k))
        self.theta2=addBiasToArray(self.theta2)
    def addRandomWeigths(self):
        #theta1= N+bias by middleNodes+bias 
        #theta2= middleNodes+bias by k
        #self.theta1=np.random.rand(self.numInputs,self.numMiddleNodes)
        #self.theta2=np.random.rand(self.numMiddleNodes,self.k)
        
        self.theta1=np.random.uniform(-0.1, 0.1, (self.numInputs,self.numMiddleNodes))
        self.theta2=np.random.uniform(-0.1, 0.1, (self.numMiddleNodes,self.k))
        #self.theta1=addBiasToArray(self.theta1)
        self.theta2=addBiasToArray(self.theta2)
    def doForwardPropForTrainingExample(self,trainingExample):
        self.a1=self.ds[trainingExample]
        self.a1=np.atleast_2d(self.a1)
        #
        #print self.a2
        self.a2=sV(np.dot(np.transpose(self.theta1),np.transpose(self.a1)))
        #print self.a2
        self.a2=addBiasToArray(self.a2)
        self.out=sV(np.dot(np.transpose(self.theta2),self.a2))
    def doForwardPropForNew(self,new):
        self.a1=new
        self.a1=np.atleast_2d(self.a1)
        #
        #print self.a2
        self.a2=sV(np.dot(np.transpose(self.theta1),np.transpose(self.a1)))
        #print self.a2
        self.a2=addBiasToArray(self.a2)
        self.out=sV(np.dot(np.transpose(self.theta2),self.a2))
    def computeErrorsForTrainingExample(self,trainingExample):
        ans=getBinaryVector(self.k,self.y[trainingExample])
        #out error
        self.eOut=(self.out-ans)
        #middle error
        sigmoidDeriratives=np.multiply(self.a2,1-self.a2)
        self.eMid=np.multiply(np.dot(self.theta2,self.eOut),sigmoidDeriratives)
        self.eMid=self.eMid[1:]
    def getDerirative(self):
        self.theta1Gradient=np.zeros((self.numMiddleNodes,self.numInputs))
        self.theta2Gradient=np.zeros((self.k,self.numMiddleNodes+1))
        
        self.theta1GradientTotal=np.zeros((self.numInputs,self.numMiddleNodes))
        self.theta2GradientTotal=np.zeros((self.numMiddleNodes+1,self.k))
        for i in range(self.numTrainingExamples):
            self.doForwardPropForTrainingExample(i)
            self.computeErrorsForTrainingExample(i)
            #
            for ii in range(self.numInputs):
                m=self.eMid*np.transpose(self.a1)[ii]
                self.theta1Gradient[:,ii]=np.transpose(m)
            self.theta1GradientTotal+=np.transpose(self.theta1Gradient)
            #
            for ii in range(self.numMiddleNodes):
                m=self.eOut*self.a2[ii]
                self.theta2Gradient[:,ii]=np.transpose(m)
            self.theta2GradientTotal+=np.transpose(self.theta2Gradient)
            #print self.theta2GradientTotal
            #print "end"
        self.theta2GradientTotal= self.theta2GradientTotal/self.numTrainingExamples
        self.theta1GradientTotal= self.theta1GradientTotal/self.numTrainingExamples
        #self.partialDerMid=self.eMidTotal/self.numTrainingExamples
        #self.partialDerOut=self.eOutTotal/self.numTrainingExamples

    def doGradientDescent(self):
        self.theta1-=self.alpha*self.theta1GradientTotal
        self.theta2-=self.alpha*self.theta2GradientTotal
    def learn(self,iterations):
        for i in range(iterations):
            self.getDerirative()
            self.doGradientDescent()
            print (self.getCost())
    def guess(self,trainingExample):
        self.doForwardPropForTrainingExample(trainingExample)
        return self.out
    def guessNew(self,new):
        self.doForwardPropForNew(new)
        return self.out
    def getCost(self):
        s=0
        for i in range(self.numTrainingExamples):
            ss=0
            y=getBinaryVector(self.k,self.y[i])
            guess=self.guess(i)
            for ii in range(self.k):
                yk=y[ii]
                if yk==1:
                    ss+=math.log(guess[ii])
                elif yk==0:
                    if guess[ii]==1:
                        ss+=math.log(0.000001)
                    else:
                        ss+=math.log(1-guess[ii])
            s+=ss
        return s/self.numTrainingExamples
    def calculateError(self):
        e=0
        for i in range(self.numTrainingExamples):
            #print self.guess(i)
            g=self.guess(i).argmax(axis=0)
            #print self.y[i],g 
            if g != self.y[i]:
                e+=1
        return e
    def calculateErrorTestSet(self,test):
        e=0
        test2=np.insert(test[0],0,1,1)
        y=test[1]
        for i in range(test2.shape[0]):
            #print self.guess(i)
            g=self.guessNew(test2[i]).argmax(axis=0)
            #print self.y[i],g 
            if g != y[i]:
                e+=1
        return e
#test
nn=NeuralNetwork(ss,small_set[1],1,10)

nn.learn(1)
print ("saving learned thetas to files t1 and t2")
np.save("t1",nn.theta1)
np.save("t2",nn.theta2)

print (nn.calculateError())
print ("guessed %d out of %d" %(nn.calculateErrorTestSet(test_set), len(test_set[0])))
#print nn.guess(0)