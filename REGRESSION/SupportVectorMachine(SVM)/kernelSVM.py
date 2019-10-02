#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:28:19 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalR
from Utils.Loss import loss
from Utils.kernels import Kernels

class kSVM(EvalR, loss, Kernels):
    '''
    Kernelized SVM via Gradient ascent.
    -----------------------------------
    Dual Lagrangian formulation
    for kernel SVMs.
    '''
    def __init__(self, kernel = None, C = None):
        self.kernel = kernel
        if not C:
            C = 1.0
            self.C = C
        return
    
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
       
    def alpha_y_i_kernel(self, X, y):
        '''
        :params: X:
        :params: y:
        '''
        alpha = np.zeros(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i alpha_j
        self.y_i_s = self.y_i(y) #y_i y_j
        self.kernel = Kernels.linear(X, X) #Using the RBF kernel to get Gram matrix
        return (alpha, self.alph_s, self.y_i_s, self.kernel)
        
    def cost(self):
        '''
        return type: x E R
        '''
        
        return np.sum(self.alpha) - .5 * np.sum(self.alpha_i_s * self.kernel * self.y_i_s )
    
    def fit(self, X, y, lr = None, iterations = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: alpha: scalar alpha value
        :params: iterations: integer iteration
        '''
        self.X = X
        self.Y = y
        if not lr:
            lr = 1e-5
            self.lr = lr
        if not iterations:
            iterations = 500
            self.iteration = iterations
        self.alpha, self.alpha_i_s, self.y_i_s,  self.kernel = self.alpha_y_i_kernel(X, y)
        cost = np.zeros(iterations)
        for ii in  range(self.iteration):
            cost[ii] = self.cost()
            print(f"Cost of computation: {cost[ii]}")
            #perform gradient ascent for maximization.
            self.alpha = self.alpha + self.lr * (np.ones(X.shape[0]) - np.dot(self.y_i_s * self.kernel, self.alpha))
            #0 < alpha < C
            self.alpha[self.alpha < 0 ] = 0
            self.alpha[self.alpha > self.C] = self.C
            
        return self
    
    def predict(self, X):
        yhat:int = np.sign(np.dot(self.alpha * self.Y, Kernels.linear(self.X, X)))
        for enum, ii in enumerate(yhat):
            if ii <0:
                yhat[enum] = 0
        return yhat
            
            
            
#%% Testing
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
kernelsvm = kSVM().fit(X_train, Y_train)
kernelsvm.predict(X_test)
np.mean(kernelsvm.predict(X_test) == Y_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = kernelsvm.predict(X_test))          
            
            
       
            