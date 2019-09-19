#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:19:27 2019

@author: kenneth
"""

import numpy as np

class loss:
    def __init__(self):
        return
    
    @staticmethod
    def sigmoid(X, beta):
        '''
        Also known as the logistic loss,
        especially because it is used 
        for logistic regression
        :params: X: traing data at ith iteration
        :return: 0 or 1
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def hinge(X, beta):
        '''
        Also known as the logistic loss,
        especially because it is used 
        for logistic regression
        :params: X: traing data at ith iteration
        :return: 0 or 1
        '''
        return np.maximum(1- np.dot(X, beta), 0)
    
    @staticmethod
    def relu(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or max
        '''
        return np.maximum(np.dot(X, beta), 0)
    
    @staticmethod
    def square(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or max
        '''
        return .5(np.dot(X, beta) + 1)
    
    @staticmethod
    def exponential(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or max
        '''
        return np.exp(2*np.dot(X, beta))/(1 + 2*np.dot(X, beta))
    
    @staticmethod
    def tanh(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or tanh(X, beta)
        '''
        return (np.exp(np.dot(X, beta)) - np.exp(-np.dot(X, beta)))/\
                (np.exp(np.dot(X, beta)) + np.exp(-np.dot(X, beta)))
                
    