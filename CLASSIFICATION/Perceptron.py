#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:34:58 2019

@author: kenneth
"""
import numpy as np


class Perceptron(object):
    def __init__(self, activation = False):
        self.activation = activation
        return
    
    @staticmethod
    def sigmoid(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or 1
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def relu(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or max
        '''
        return np.max(np.dot(X, beta), 0)
    
    @staticmethod
    def tanh(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or -beta/x
        '''
        return np.tanh(np.dot(X, beta))
    
    @staticmethod
    def cost(X, Y, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or 1
        '''
        return -(1/len(Y)) * np.sum((Y*np.log(Perceptron.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - Perceptron.sigmoid(X, beta))))
    
    def fit(self, X, Y, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.beta = np.zeros(X.shape[1]).reshape(-1, 1)
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        if not self.activation:
            for ii in range(self.iterations):
                self.beta = self.beta + (1/len(Y)) *(self.alpha) * X.T.dot(Y - Perceptron.sigmoid(X, self.beta))
                self.beta_rec[ii, :] = self.beta.T
                self.cost_rec[ii] = self.cost(X, Y, self.beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
            return self
        elif self.activation == 'relu':
            for ii in range(self.iterations):
                self.beta = self.beta + (1/len(Y)) *(self.alpha) * X.T.dot(Y - Perceptron.relu(X, self.beta))
                self.beta_rec[ii, :] = self.beta.T
                self.cost_rec[ii] = self.cost(X, Y, self.beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
            return self
        elif self.activation == 'tanh':
            for ii in range(self.iterations):
                self.beta = self.beta + (1/len(Y)) *(self.alpha) * X.T.dot(Y - Perceptron.tanh(X, self.beta))
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
        if not self.activation:
            for ii in range(len(y_pred)):
                if Perceptron.sigmoid(X[ii], self.beta) >= 0.5:
                    y_pred[ii] = 1
                elif Perceptron.sigmoid(X[ii], self.beta) < 0:
                    y_pred[ii] = 0
            return y_pred
        elif self.activation == 'relu':
            print('relu')
            for ii in range(len(y_pred)):
                if Perceptron.sigmoid(X[ii], self.beta) > 0:
                    y_pred[ii] = 1
                elif Perceptron.sigmoid(X[ii], self.beta) < 0:
                    y_pred[ii] = 0
            return y_pred
        elif self.activation == 'tanh':
            for ii in range(len(y_pred)):
                if Perceptron.sigmoid(X[ii], self.beta) > 0:
                    y_pred[ii] = 1
                elif Perceptron.sigmoid(X[ii], self.beta) < 0:
                    y_pred[ii]
            return  y_pred
        
        
#%%

pctron = Perceptron(activation='relu').fit(X_train, Y_train.reshape(-1, 1), 0.1, 100)
pctron.predict(X_test)

















