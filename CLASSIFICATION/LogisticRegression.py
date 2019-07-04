#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:25:46 2019

@author: kenneth
"""
import numpy as np

class Logistic():
    def __init__(self):
        return
    
    #classification metrics
    '''
                            Actual
                        +ve       -ve
                    ---------------------
                +ve |   TP    |     FP  | +---> Precision
                    --------------------- |
    predicted   -ve |   FN    |   TN    | v
                    --------------------- Recall
    '''
    def TP(self, A, P):
        '''Docstring
        when actual is 1 and prediction is 1
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 1) & (P == 1))
    
    def FP(self, A, P):
        '''Docstring
        when actual is 0 and prediction is 1
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 0) & (P == 1))
    
    def FN(self, A, P):
        '''Docstring
        when actual is 1 and prediction is 0
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 1) & (P == 0))
    
    def TN(self, A, P):
        '''Docstring
        when actual is 0 and prediction is 0
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 0) & (P == 0))
    
    def confusionMatrix(self, A, P):
        '''Docstring
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        TP, FP, FN, TN = (self.TP(A, P),\
                          self.FP(A, P),\
                          self.FN(A, P),\
                          self.TN(A, P))
        return np.array([[TP, FP], [FN, TN]])
    
    def accuracy(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return (self.TP(A, P) + self.TN(A, P))/(self.TP(A, P) + self.FP(A, P) +\
                                                self.FN(A, P) + self.TN(A, P))
    
    def precision(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return self.TP(A, P)/(self.TP(A, P) + self.FP(A, P))
    
    def recall(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return self.TP(A, P)/(self.TP(A, P) + self.FN(A, P))
    
    def f1(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return (2 * (self.precision(A, P) * self.recall(A, P)))/(self.precision(A, P) + self.recall(A, P))
    
    def summary(self, A, P):
        print('*'*40)
        print('\t\tSummary')
        print('*'*40)
        print('>> Accuracy: %s'%self.accuracy(A, P))
        print('>> Precision: %s'%self.precision(A, P))
        print('>> Recall: %s'%self.recall(A, P))
        print('>> F1-score: %s'%self.f1(A, P))
        print('*'*40)
        
    @staticmethod
    def sigmoid(X, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: beta: weights N x 1
        
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def cost(X, Y, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: Y: label y \in {0,1} N x 1 dimension
        :params: beta: weights N x 1
        
        '''
        return -(1/len(Y)) * np.sum((Y*np.log(Logistic.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - Logistic.sigmoid(X, beta))))
    
    def fit(self, X, Y, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        for ii in range(self.iterations):
            #compute gradient
            self.beta = self.beta + (1/len(Y)) *(self.alpha) * X.T.dot(Y - Logistic.sigmoid(X, self.beta))
            self.beta_rec[ii, :] = self.beta.T
            self.cost_rec[ii] = self.cost(X, Y, self.beta)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        for ii in range(len(y_pred)):
            if Logistic.sigmoid(X[ii], self.beta) > 0.5:
                y_pred[ii] = 1
        return y_pred


class stochasticLogistic(Logistic):
    def __init__(self, alpha, iterations):
        super().__init__()
        self.alpha = alpha
        self.iterations = iterations
        return
           
    @staticmethod
    def sigmoid(X, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: beta: weights N x 1
        
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def cost(X, Y, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: Y: label y \in {0,1} N x 1 dimension
        :params: beta: weights N x 1
        
        '''
        return -(1/len(Y)) * np.sum((Y*np.log(stochasticLogistic.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - stochasticLogistic.sigmoid(X, beta))))
    
    def fit(self, X, Y):
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        ylen = len(Y)
        for ii in range(self.iterations):
            #compute stochastic gradient
            sampledCost = []
            for ij in range(ylen):
                random_samples = np.random.randint(1, ylen)
                X_samp = X[:random_samples]
                Y_samp = Y[:random_samples]
                self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * X_samp.T.dot(Y_samp - stochasticLogistic.sigmoid(X_samp, self.beta))
                self.beta_rec[ii, :] = self.beta.T
                sampledCost.append(self.cost(X_samp, Y_samp, self.beta))
            self.cost_rec[ii] = np.average(sampledCost)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        for ii in range(len(y_pred)):
            if stochasticLogistic.sigmoid(X[ii], self.beta) > 0.5:
                y_pred[ii] = 1
        return y_pred

class minibatchLogistic(Logistic):
    def __init__(self, alpha, iterations):
        super().__init__()
        self.alpha = alpha
        self.iterations = iterations
        return
           
    @staticmethod
    def sigmoid(X, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: beta: weights N x 1
        
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def cost(X, Y, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: Y: label y \in {0,1} N x 1 dimension
        :params: beta: weights N x 1
        
        '''
        return -(1/len(Y)) * np.sum((Y*np.log(minibatchLogistic.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - minibatchLogistic.sigmoid(X, beta))))
    
    def fit(self, X, Y, batchSize = None):
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        ylen = len(Y)
        batchNumber = int(ylen/batchSize)
        for ii in range(self.iterations):
            #compute minibatch gradient
            sampledCost = []
            random_samples = np.random.permutation(ylen)
            X_random = X[random_samples]
            Y_random = Y[random_samples]
            for ij in range(0, ylen, batchNumber):
                X_samp = X_random[ij:ij+batchSize]
                Y_samp = Y_random[ij:ij+batchSize]
                self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * X_samp.T.dot(Y_samp - minibatchLogistic.sigmoid(X_samp, self.beta))
                self.beta_rec[ii, :] = self.beta.T
                sampledCost.append(self.cost(X_samp, Y_samp, self.beta))
            self.cost_rec[ii] = np.average(sampledCost)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        for ii in range(len(y_pred)):
            if minibatchLogistic.sigmoid(X[ii], self.beta) > 0.5:
                y_pred[ii] = 1
        return y_pred
    
#%%
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(n_samples=100, centers=2, n_features=2 )
X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
logit = Logistic().fit(X_train, Y_train, 0.1, 100)
y_pred = logit.predict(X_test)
logit.summary(Y_test, y_pred)
logit.confusionMatrix(Y_test, y_pred)

stlog = stochasticLogistic(alpha=0.1, iterations=100).fit(X_train, Y_train)
y_pred = stlog.predict(X_test)
stlog.summary(Y_test, y_pred)
stlog.confusionMatrix(Y_test, y_pred)

minilog = minibatchLogistic(alpha=0.1, iterations=100).fit(X_test, Y_test, batchSize= 10)
y_pred = minilog.predict(X_test)
minilog.summary(Y_test, y_pred)
minilog.confusionMatrix(Y_test, y_pred)