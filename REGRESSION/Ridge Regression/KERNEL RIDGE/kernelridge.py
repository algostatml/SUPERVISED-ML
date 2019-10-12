#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:55:01 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalR
from Utils.Loss import loss
from Utils.kernels import Kernels

class kernelridge(EvalR, loss, Kernels):
    def __init__(self, kernel = None, lamda = None):
        super().__init__()
        if not kernel:
            kernel = 'linear'
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not lamda:
            lamda = 100000
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
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
        
    def fit(self, X, y):
        '''
        :param: X: NxD
        :param: Dx1
        '''
        self.X = X
        self.y = y
        self.alpha = np.linalg.solve(self.kernelize(self.X, self.X) + self.lamda*np.eye(self.X.shape[0]), self.y)
        return self
    
    def predict(self, X):
        '''
        :param: X: NxD
        :return type: Dx1 vector
        '''
        return np.dot((self.kernelize(self.X, X).T * self.y), self.alpha.T)
    
    
#%% Testing

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler

X, y = load_boston().data, load_boston().target
X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .3)
kridge = kernelridge().fit(X_train, Y_train)
kridge.predict(X_test)
kridge.summary(X, Y_test, kridge.predict(X_test))
    
#%%

from sklearn.kernel_ridge import KernelRidge
clf = KernelRidge(alpha=1.0, kernel='linear')    
clf.fit(X, y)
kridge.summary(X, Y_test, clf.predict(X_test))
