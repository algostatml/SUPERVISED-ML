#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:11:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class Kernels:
    '''Docstring
    Kernels are mostly used for solving
    non-lineaar problems. By projecting/transforming
    our data into a subspace, making it easy to
    almost accurately classify our data as if it were
    still in linear space.
    '''
    def __init__(self):
        return
    
    @staticmethod
    def linear(x1, x2):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        return x1.dot(x2.T)
    
    @staticmethod
    def rbf(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1.0/x1.shape[1]
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 1)**2)
        
    @staticmethod
    def sigmoid(x1, x2, gamma = None, C = None):
        '''
        logistic or sigmoid kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = .05
        if not C:
            C = 1
        return np.tanh(gamma * x1.dot(x2.T))
    
    @staticmethod
    def polynomial(x1, x2, d = None):
        '''
        polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: d: polynomial degree
        :return type: kernel(Gram) matrix
        '''
        if not d:
            d = 2
        return (x1.T.dot(x2))**d
    
    @staticmethod
    def cosine(x1, x2):
        '''
        Cosine kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        
        return (x1.T.dot(x2)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1))
    
    @staticmethod
    def correlation(x1, x2, gamma = None):
        '''
        Correlation kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1.0/x1.shape[1]
        return np.exp((x1.T.dot(x2)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1)) - gamma)
    
    
    