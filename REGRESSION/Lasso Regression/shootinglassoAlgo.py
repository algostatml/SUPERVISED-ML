#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:49:38 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalR


class Shootinglasso(EvalR):
    '''Docstring
    Coordinate descent for Lasso (Shooting algorithm) - Wenjiang Fu (1998)
    ------------------------
    lasso regression via gradient descent
    Reference: http://www.hpc.unm.edu/~andriese/doc/ref1_fu.pdf
    --------------------------------------------
    Regularization via lasso regression.
    l1-norm induces sparsity by <shrinking>
    the coefficients of the matrix to zero.
    '''
    def __init__(self, lamda:float = None):
        super().__init__()
        self.lamda:float = lamda
        if not lamda:
            lamda:float = 0.01
            self.lamda:float = lamda
        else:
            self.lamda:float = lamda
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1, ...,bn
        Return: cost
        '''
        return (1/2*len(Y)) * (np.sum(np.square(X.dot(beta) - Y)) + (self.lamda*np.sum(np.absolute(beta))))
        
        
    def fit(self, X, Y, alpha = None, iterations = None, shooting: bool = True):
        '''
        :param: X = NxD feature matrix
        :param: Y = Dx1 column vector
        :param: beta = Dx1 beta vector coefficients. Default is zero vector.
        :param: alpha = learning rate. Default 1e-2. Default is 0.1
        :param: iterations = number of iterations. Default is 100.
        
        Return type: final beta/coefficients, cost and bata iterations
        '''
        if not alpha:
            '''
            Alpha is huge because we are testing
            it on the boston dataset.
            '''
            alpha = 2.1
            self.alpha = alpha
        else:
            self.alpha = alpha
        
        if not iterations:
            iterations = 10000
            self.iterations = iterations
        else:
            self.iterations = iterations
        
        self.alpha = alpha
        if shooting == False:
            self.beta = np.zeros(X.shape[1])
        else:
            self.beta = np.linalg.solve(self.lamda*np.eye(X.shape[1]) + X.T.dot(X), X.T.dot(Y))
        self.cost_rec = np.zeros(iterations)
        self.beta_rec = np.zeros((iterations, X.shape[1]))
        for self.iter in range(self.iterations):
            #compute gradient
            for i in range(X.shape[1]):
                s_0 = 2 * np.dot(X[:, i].T, X[:, i])
                s_j = 2 * np.dot(X[:, i], Y - np.dot(X, self.beta) + self.beta[i]*X[:, i])
                if s_0 == 0:
                    self.beta[i] = 0
                elif s_j < -self.lamda:
                    self.beta[i] = (s_j + self.lamda)/s_0
                elif s_j > self.lamda:
                    self.beta[i] = (s_j - self.lamda)/s_0
                else:
                    self.beta[i] = 0
            self.beta_rec[self.iter, :] = self.beta.T
            self.cost_rec[self.iter] = self.cost(X, Y, self.beta)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))
        print('*'*40)
        return self
    
    #prediction
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    #plot cost
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(self.iterations), self.cost_rec)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
#%% Testing

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler

X, y = load_boston().data, load_boston().target
X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .3)
slasso = Shootinglasso().fit(X_train, Y_train, shooting=False)
slasso.summary(X, Y_test, slasso.predict(X_test))
slasso.plot_cost()

#%% visualize coefficients

import matplotlib.pyplot as plt
import pandas as pd
pd.Series(slasso.beta).plot(kind = 'bar')
