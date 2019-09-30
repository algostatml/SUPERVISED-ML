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

class linearSVM(object):
    def __init__(self, C = None):
        
        if not C:
            C = 0.1
            self.C = C
        else:
            self.C = C
        return
    
    def cost(self, X, y, beta):
        '''
        Hinge loss function
        '''
        return .5 * beta.dot(beta) + self.C * np.sum(loss.hinge(X, y, beta))
    
    def fit(self, X, y, alpha, iterations):
        if not alpha:
            alpha = 0.01
            self.alpha = alpha
        else:
            self.alpha = alpha
        
        if not iterations:
            iterations = 1000
            self.iterations = iterations
        else:
            self.iterations = iterations
            
        