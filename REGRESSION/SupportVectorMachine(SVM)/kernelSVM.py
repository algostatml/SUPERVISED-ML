#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:28:19 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class kDualSVM(EvalC, loss, Kernels):
    '''
    Kernelized SVM via Gradient ascent.
    -----------------------------------
    Dual Lagrangian formulation
    for kernel SVMs.
    '''
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf' #default
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not C:
            C = 1.0
            self.C = C
        else:
            self.C = C
        return
    
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
       
    def kernelize(self, x1, x2):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2)
        
        
    def alpha_y_i_kernel(self, X, y):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.zeros(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.y_i_s = self.y_i(y) #y_i's y_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.y_i_s, self.k)
        
    def cost(self):
        '''
        return type: x E R
        '''
        return np.dot(self.alpha, np.ones(self.X.shape[0])) - .5 * np.sum(self.alpha_i_s * self.knl * self.y_i_s )
    
    def fit(self, X, y, lr:float = None, iterations:int = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: lr: scalar learning rate value
        :params: iterations: integer iteration
        '''
        self.X = X
        self.Y = y
        if not lr:
            lr = 1e-5
            self.lr = lr
        else:
            self.lr = lr
        if not iterations:
            iterations = 500
            self.iteration = iterations
        else:
            self.iteration = iterations
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(self.X, self.Y)
        cost = np.zeros(iterations)
        for ii in  range(self.iteration):
            cost[ii] = self.cost()
            print(f"Cost of computation: {cost[ii]}")
            #perform gradient ascent for maximization.
            self.alpha = self.alpha + self.lr * (np.ones(X.shape[0]) - np.dot(self.y_i_s * self.knl, self.alpha))
            #0 < alpha < C
            self.alpha[self.alpha < 0] = 0
            self.alpha[self.alpha > self.C] = self.C
            
        return self
    
    def predict(self, X):
        yhat:int = np.sign(np.dot(self.alpha * self.Y, self.kernelize(self.X, X)))
        for enum, ii in enumerate(yhat):
            if ii <=0:
                yhat[enum] = 0
        return yhat
            
#%%
class kprimalSVM(EvalC, loss, Kernels):
    '''
    Kernelized SVM via Gradient ascent.
    -----------------------------------
    Primal function tranformed to dual formulation
    for kernel SVMs.
    '''
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf' #default
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not C:
            C = 1.0
            self.C = C
        else:
            self.C = C
        return
    
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
       
    def kernelize(self, x1, x2):
        '''
        :params: X: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2)
        
        
    def alpha_y_i_kernel(self, X, y):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.zeros(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.y_i_s = self.y_i(y) #y_i's y_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.y_i_s, self.k)
        
    def hingeloss(self):
        '''
        Hinge loss for the dual primal
        '''
        return np.maximum(0, 1 - self.y_i_s * self.alpha * self.kernelize(self.X, self.X))
    
    def cost(self):
        '''
        return type: x E R
        '''
        return np.sum(self.alpha_i_s * self.knl * self.y_i_s ) + self.C * np.sum(self.hingeloss())
    
    def fit(self, X, y, lr:float = None, iterations:int = None):
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
        else:
            self.lr = lr
        if not iterations:
            iterations = 500
            self.iteration = iterations
        else:
            self.iteration = iterations
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(X, y)
        cost = np.zeros(iterations)
        for ii in  range(self.iteration):
            cost[ii] = self.cost()
            print(f"Cost of computation: {cost[ii]}")
            #perform gradient ascent for maximization.
            self.alpha = self.alpha - self.lr * (np.dot(self.y_i_s * self.knl, self.alpha) - np.dot(self.Y * self.knl, self.Y))
            #0 < alpha < C
            self.alpha[self.alpha < 0 ] = 0
            self.alpha[self.alpha > self.C] = self.C
            
        return self
    
    def predict(self, X):
        yhat:int = np.sign(np.dot(self.alpha * self.Y, self.kernelize(self.X, X)))
        for enum, ii in enumerate(yhat):
            if ii <0:
                yhat[enum] = 0
        return yhat
    
    
#%% Testing
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(n_samples=1000, centers=2, n_features=2)
X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
kernelsvm = kDualSVM(kernel='linear').fit(X_train, Y_train)
kernelsvm.predict(X_test)
kernelsvm.summary(Y_test, kernelsvm.predict(X_test), kernelsvm.alpha)
plt.scatter(X_test[:, 1], X_test[:, 2], c = kernelsvm.predict(X_test))          


primalkernelsvm = kprimalSVM(kernel='linear').fit(X_train, Y_train)
primalkernelsvm.predict(X_test)
primalkernelsvm.summary(Y_test, primalkernelsvm.predict(X_test), primalkernelsvm.alpha)
plt.scatter(X_test[:, 1], X_test[:, 2], c = primalkernelsvm.predict(X_test))          
            
       
            