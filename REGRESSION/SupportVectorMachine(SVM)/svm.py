#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:51:41 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalR
from Utils.Loss import loss


class linearSVM(loss):
    def __init__(self, C = None):
        '''
        Linear SVM via Gradient descent
        :params: C: misclassification penalty. 
                    Default value is 0.1.
        '''
        if not C:
            C = 1.0
            self.C = C
        else:
            self.C = C
        return
    
    def cost(self, X, y, beta):
        '''
        Hinge loss function
        ------------------
        :params: X: feature space
        :params: y: target
        :params: beta: weights parameters.
        '''
        return 0.5 * beta.dot(beta) + self.C * np.sum(loss.hinge(X, y, beta))
    
    def margins(self, X, y, beta):
        '''
        :param: X: NxD
        :param: y: Nx1
        :param: beta: Dx1
        '''
        return y*np.dot(X, beta)
        
    def fit(self, X, y, alpha = None, iterations = None):
        if not alpha:
            alpha = 1e-5
            self.alpha = alpha
        else:
            self.alpha = alpha
        if not iterations:
            iterations = 500
            self.iterations = iterations
        else:
            self.iterations = iterations
        self.beta = np.random.randn(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        for _ in range(self.iterations):
            self.margin = self.margins(X, y, self.beta)
            #adjust parameters according to misclafication
            indices = np.where(self.margin < 0)
            self.beta = self.beta - self.alpha * self.C * y[indices].dot(X[indices])
            print(f"cost is {self.cost(X, y, self.beta)}")
        return self
    
    def predict(self, X):
        yhat:int = np.zeros(X.shape[0])
        for enum, ii in enumerate(np.sign(X.dot(self.beta))):
            if ii >0:
                yhat[enum] = 1
        return yhat
    

class StochasticlinearSVM(loss):
    def __init__(self, C = None):
        '''
        Linear SVM via Stochastic Gradient descent
        :params: C: misclassification penalty. 
                    Default value is 0.1.
        '''
        if not C:
            C = 1.0
            self.C = C
        else:
            self.C = C
        return
    
    def cost(self, X, y, beta):
        '''
        Hinge loss function
        ------------------
        :params: X: feature space
        :params: y: target
        :params: beta: weights parameters.
        '''
        return 0.5 * beta.dot(beta) + self.C * np.sum(loss.hinge(X, y, beta))
    
    def margins(self, X, y, beta):
        '''
        :param: X: NxD
        :param: y: Nx1
        :param: beta: Dx1
        '''
        return y*np.dot(X, beta)
        
    def fit(self, X, y, alpha = None, iterations = None):
        if not alpha:
            alpha = 1e-5
            self.alpha = alpha
        else:
            self.alpha = alpha
        if not iterations:
            iterations = 500
            self.iterations = iterations
        else:
            self.iterations = iterations
        self.beta = np.random.randn(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        ylen = len(y)
        for _ in range(self.iterations):
            for ij in range(ylen):
                random_samples = np.random.randint(1, ylen)
                X_samp = X[:random_samples]
                y_samp = y[:random_samples]
                self.margin = self.margins(X_samp, y_samp, self.beta)
                #adjust parameters according to misclafication
                indices = np.where(self.margin < 0)
                self.beta = self.beta - self.alpha * self.C * y_samp[indices].dot(X_samp[indices])
                print(f"cost is {self.cost(X, y, self.beta)}")
        return self
    
    def predict(self, X):
        yhat:int = np.zeros(X.shape[0])
        for enum, ii in enumerate(np.sign(X.dot(self.beta))):
            if ii >0:
                yhat[enum] = 1
        return yhat
              
#%%
import matplotlib.pyplot as plt
lsvm = linearSVM().fit(X_train, Y_train)
lsvm.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = lsvm.predict(X_test))
np.mean(lsvm.predict(X_test) == Y_test)


slsvm = StochasticlinearSVM().fit(X_train, Y_train)
slsvm.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = slsvm.predict(X_test))
np.mean(slsvm.predict(X_test) == Y_test)
