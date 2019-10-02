#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:21:48 2019

@author: kenneth
"""
from __future__ import absolute_import, division
import numpy as np
from Utils.utils import EvalR


class etaLasso(EvalR):
    def __init__(self, eta:float = None, lamda:float = None):
        '''
        :param: eta: scalar value
        :param: lamda: scalar value
        '''
        if not eta:
            eta = 100
            self.eta = eta
        else:
            self.eta = eta
        
        if not lamda:
            lamda = 1e-5
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    @staticmethod
    def inverse(mat):
        '''
        :param: matrix NxD or NxN
        '''
        return np.linalg.inv(mat) 
    
    def fit(self, X, y):
        '''
        :param: X: NxD feature space
        :param: y = Dx1 target
        '''
        self.beta = np.linalg.solve((X.T.dot(X) + self.lamda * etaLasso.inverse(self.eta*np.eye(X.shape[1]))), X.T.dot(y))
        return self
    
    def predict(self, X):
        '''
        :param: X: NxD test data
        '''
        return X.dot(self.beta)
    

class etaGD(EvalR):
    def __init__(self, eta:float = None, lamda:float = None):
        '''
        :param: eta: scalar value
        :param: lamda: scalar value
        '''
        if not eta:
            eta = 1000000000
            self.eta = eta
        else:
            self.eta = eta
        
        if not lamda:
            lamda = 1e-7
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    @staticmethod
    def inverse(mat):
        '''
        :param: matrix NxD or NxN
        '''
        return np.linalg.inv(mat) 
    
    def cost(self, X, y, beta):
        '''
        :param: X: NxD feature space
        :param: y = Dx1 target
        '''
        return (1/2*(len(y)))*np.sum(np.square(y - (X.dot(beta)))) + ((self.lamda/2) * (beta.dot(beta)/self.eta + self.eta))
        
    
    def fit(self, X, y, alpha:float = None, iterations:int = None):
        if not alpha:
            alpha = .001
            self.alpha = alpha
        else:
            self.alpha = alpha
        if not iterations:
            iterations = 500000
            self.iterations = iterations
        else:
            self.iterations = iterations
        
        self.beta = np.zeros(X.shape[1])
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        self.cost_rec = np.zeros(self.iterations)
        for ii in range(self.iterations):
            self.beta = self.beta - 1/len(y) * ((X.T.dot(X.dot(self.beta) - y)) + (self.lamda*self.beta)/self.eta)
            self.beta_rec[ii, :] = self.beta.T
            self.cost_rec[ii] = self.cost(X, y, self.beta)
            print('-'*40)
            print(f"Cost of computation: {self.cost(X, y, self.beta)}")
        return self
    
    def predict(self, X):
        '''
        param: X: NxD test data
        '''
        return X.dot(self.beta)
            
class etaSGD(EvalR):
    def __init__(self, eta:float = None, lamda:float = None):
        '''
        :param: eta: scalar value
        :param: lamda: scalar value
        '''
        if not eta:
            eta = 1000000000
            self.eta = eta
        else:
            self.eta = eta
        
        if not lamda:
            lamda = 1e-7
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    @staticmethod
    def inverse(mat):
        '''
        :param: matrix NxD or NxN
        '''
        return np.linalg.inv(mat) 
    
    def cost(self, X, y, beta):
        '''
        :param: X: NxD feature space
        :param: y = Dx1 target
        '''
        return (1/2*(len(y)))*np.sum(np.square(y - (X.dot(beta)))) + ((self.lamda/2) * (beta.dot(beta)/self.eta + self.eta))
        
    
    def fit(self, X, y, alpha:float = None, iterations:int = None):
        if not alpha:
            alpha = .001
            self.alpha = alpha
        else:
            self.alpha = alpha
        if not iterations:
            iterations = 10000
            self.iterations = iterations
        else:
            self.iterations = iterations
        
        self.beta = np.zeros(X.shape[1])
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        self.cost_rec = np.zeros(self.iterations)
        for ii in range(self.iterations):
            for ij in range(len(y)):
                random_samples = np.random.randint(1, len(y))
                X_samp = X[:random_samples]
                Y_samp = y[:random_samples]
                self.beta = self.beta - 1/len(Y_samp) * ((X_samp.T.dot(X_samp.dot(self.beta) - Y_samp)) + (self.lamda*self.beta)/self.eta)
                self.beta_rec[ii, :] = self.beta.T
                self.cost_rec[ii] += self.cost(X_samp, Y_samp, self.beta)
                print('-'*40)
                print(f"Cost of computation: {self.cost(X, y, self.beta)}")
        return self
    
    def predict(self, X):
        '''
        param: X: NxD test data
        '''
        return X.dot(self.beta)          
    
#%% Testing etalasso closed form

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

X, y = load_boston().data, load_boston().target
X = Normalizer().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .3)
etaregressor = etaLasso().fit(X_train, Y_train)
pred = etaregressor.predict(X_test)
etaregressor.summary(X, Y_test, pred)

#%% By gradient descent

etagd = etaGD().fit(X_train, Y_train)
epred = etagd.predict(X_test)
etagd.summary(X, Y_test, epred)

plt.plot(np.arange(etagd.iterations), etagd.cost_rec)

#%% By stochastic gradient descent

etasgd = etaSGD().fit(X_train, Y_train)
espred = etagd.predict(X_test)
etasgd.summary(X, Y_test, espred)

plt.plot(np.arange(etasgd.iterations), etasgd.cost_rec)

