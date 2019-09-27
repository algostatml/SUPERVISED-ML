#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 07:25:54 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalR


class Ridge(EvalR):
    '''Docstring
    Ridge regression via Closed form
    \beta := (XtX + \lamda * I)XtY
    --------------------------------------
    Regularization via Ridge regression.
    l2-norm reduces the risk of overfitting
    here but does not induce sparsity
    '''
    def __init__(self, lamda = None):
        super().__init__()
        self.lamda = lamda
        if not lamda:
            lamda = 0.01
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        #--Closed form
        self.beta = np.linalg.solve(self.lamda*np.eye(self.X.shape[1]) + self.X.T.dot(self.X), self.X.T.dot(self.Y))
        return self
        
    def predict(self, X):
        Y_hat = X.dot(self.beta)
        return Y_hat
    
class RidgeGD(EvalR):
    '''
    Ridge regression via Gradient Descent
    '''
    def __init__(self, lamda = None):
        super().__init__()
        self.lamda = lamda
        if not lamda:
            lamda = 0.01
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1, ..., bn
        Return: cost
        '''
        return (1/2*len(Y)) * (np.sum(np.square(X.dot(beta) - Y)) + (self.lamda*np.sum(np.square(beta))))
    
    def RGD(self, X, Y, beta, alpha, iterations):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        self.beta_rec = np.zeros((iterations, X.shape[1]))
        for self.iter in range(iterations):
            #compute gradient
            self.beta = self.beta - (1/len(Y)) *(alpha) * ((np.dot(X.T, (np.dot(X, self.beta) - Y))) + (self.lamda*self.beta))
            self.beta_rec[self.iter, :] = self.beta.T
            self.cost_rec[self.iter] = self.cost(X, Y, self.beta)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))
        print('*'*40)
        return self
            
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), self.cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
        
#%% Testing
    
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

X, y = load_boston().data, load_boston().target
X = Normalizer().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .3)
ridge = Ridge().fit(X_train, Y_train)
ridge.summary(Y_test, ridge.predict(X_test))

        
